# GPU Cluster Slow Node Detection

## 目录

- [概述](#概述)
- [检测原理](#检测原理)
- [工具说明](#工具说明)
- [使用指南](#使用指南)
- [性能基线](#性能基线)
- [故障排查](#故障排查)
- [最佳实践](#最佳实践)
- [案例分析](#案例分析)

---

## 概述

### 什么是慢节点

在GPU集群中，**慢节点（Slow Node）** 是指性能显著低于集群平均水平的节点。慢节点会严重影响分布式训练性能，因为：

1. **同步等待**：分布式训练需要等待最慢的节点完成计算
2. **通讯瓶颈**：慢节点的通讯性能会拖累整个集群
3. **资源浪费**：其他节点空闲等待，降低整体GPU利用率

### 慢节点的常见原因

#### 节点内部问题
- **NVLink故障或降速**：NVLink连接断开或运行在降速模式
- **PCIe链路问题**：PCIe运行在低于预期的代数或宽度（例如Gen3而非Gen4，x8而非x16）
- **GPU硬件故障**：GPU温度过高、频率降低、ECC错误
- **内存带宽降级**：GPU内存运行在降速模式

#### 跨节点问题
- **网络连接问题**：InfiniBand或RoCE连接不稳定
- **交换机端口故障**：叶交换机或spine交换机端口性能下降
- **网络拥塞**：网络带宽被其他任务占用
- **驱动或固件版本不匹配**：NCCL、网络驱动或固件版本不一致

### 检测方法论

本工具集采用业界最佳实践，结合：

1. **节点内部带宽检测**：检测NVLink和PCIe带宽
2. **跨节点NCCL通讯测试**：多次运行all-reduce测试获取统计数据
3. **成对测试（Pairwise Testing）**：检测每对节点之间的通讯性能
4. **二分搜索（Binary Search）**：快速定位问题节点
5. **统计分析**：计算均值、标准差、最小值、最大值，识别异常值

---

## 检测原理

### 1. 节点内部带宽检测

#### NVLink检测

NVLink是NVIDIA GPU之间的高速互连技术：

| GPU型号 | NVLink版本 | 单GPU总带宽 | 每链路带宽 |
|---------|-----------|------------|-----------|
| V100 SXM2 | NVLink 2.0 | 300 GB/s | 25 GB/s |
| A100 SXM4 | NVLink 3.0 | 600 GB/s | 50 GB/s |
| H100 SXM5 | NVLink 4.0 | 900 GB/s | 50 GB/s |

**检测方法**：
1. 使用 `nvidia-smi nvlink --status` 检查NVLink链路状态
2. 使用 `p2pBandwidthLatencyTest` 或 `nvbandwidth` 测量GPU间带宽
3. 对比实测带宽与理论基线，识别降速连接

**异常示例**：
```bash
# 正常情况：8个A100 GPU，每对GPU通过NVLink连接
GPU 0 <-> GPU 1: 280 GB/s (预期: 300 GB/s) ✓
GPU 0 <-> GPU 2: 120 GB/s (预期: 300 GB/s) ✗ 慢连接！

# 可能原因：NVLink cable松动或故障
```

#### PCIe检测

PCIe是GPU与主机之间的连接：

| PCIe代数 | x16带宽（单向） | x16带宽（双向） |
|----------|---------------|----------------|
| Gen3 | ~16 GB/s | ~32 GB/s |
| Gen4 | ~32 GB/s | ~64 GB/s |
| Gen5 | ~64 GB/s | ~128 GB/s |

**检测方法**：
1. 检查PCIe链路代数和宽度：`nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current`
2. 使用 `bandwidthTest` 测量Host-to-Device和Device-to-Host带宽
3. 识别运行在降速模式的GPU

**异常示例**：
```bash
# GPU运行在PCIe Gen3 x8（预期Gen4 x16）
GPU 0: Gen3 x8, HtoD: 8 GB/s ✗ 应该是Gen4 x16, ~31 GB/s

# 可能原因：PCIe槽位配置错误或主板问题
```

### 2. 跨节点NCCL通讯检测

#### NCCL All-Reduce测试

NCCL（NVIDIA Collective Communications Library）是GPU集群通讯库，all-reduce是最常用的集合通讯操作。

**性能指标**：
- **Algorithm Bandwidth (algbw)**：算法带宽，单个GPU的平均带宽
- **Bus Bandwidth (busbw)**：总线带宽，考虑了通讯模式后的有效带宽
- **基准**：通常应达到理论带宽的92%

**NCCL测试示例**：
```bash
# 运行all_reduce_perf测试
mpirun -np 16 --hostfile hosts \
  /usr/local/nccl-tests/build/all_reduce_perf \
  -b 8 -e 8G -f 2 -g 1

# 输出示例（8G消息）：
#       size     count    type   redop    time   algbw   busbw
#  8589934592   1048576   float     sum   22.45  382.37  250.2

# busbw: 250.2 GB/s （对于8个A100 GPU通过NVLink，预期~250 GB/s）
```

#### 多次迭代统计

为了消除偶然因素，每个测试运行多次（默认10次）：

```
迭代1: 248.5 GB/s
迭代2: 251.2 GB/s
迭代3: 249.8 GB/s
...
迭代10: 250.5 GB/s

统计分析：
  均值: 250.1 GB/s
  标准差: 1.2 GB/s
  最小值: 248.5 GB/s
  最大值: 251.8 GB/s
```

**性能判断**：
- 均值低于预期的90%：性能不达标
- 标准差过大（>5%）：性能不稳定

#### 成对测试（Pairwise Testing）

测试每对节点之间的通讯性能：

```
节点对测试矩阵：
  node1 <-> node2: 245 GB/s ✓
  node1 <-> node3: 180 GB/s ✗ 慢连接
  node1 <-> node4: 248 GB/s ✓
  node2 <-> node3: 175 GB/s ✗ 慢连接
  node2 <-> node4: 250 GB/s ✓
  node3 <-> node4: 178 GB/s ✗ 慢连接

分析：node3 在3个成对测试中均表现慢 → node3是问题节点
```

#### 二分搜索（Binary Search）

用于快速定位问题节点（适用于4+节点）：

```
步骤1：测试所有8个节点
  结果：210 GB/s（预期250 GB/s）✗ 存在慢节点

步骤2：二分为两组（节点1-4，节点5-8）
  组1（节点1-4）: 248 GB/s ✓ 正常
  组2（节点5-8）: 180 GB/s ✗ 存在慢节点

步骤3：继续二分组2（节点5-6，节点7-8）
  组2-1（节点5-6）: 245 GB/s ✓ 正常
  组2-2（节点7-8）: 120 GB/s ✗ 存在慢节点

步骤4：识别问题节点
  节点7: 测试正常
  节点8: ✗ 问题节点

结论：节点8是慢节点
```

---

## 工具说明

### 1. intra_node_bandwidth_check.sh

**功能**：检测单个节点内部的GPU带宽性能

**依赖**：
- nvidia-smi
- bandwidthTest (CUDA Samples)
- p2pBandwidthLatencyTest (CUDA Samples)
- nvbandwidth (可选，CUDA Samples新版本)

**用法**：
```bash
# 基本用法
./intra_node_bandwidth_check.sh -o ./results

# 自定义阈值
./intra_node_bandwidth_check.sh -o ./results -t 85

# 详细输出
./intra_node_bandwidth_check.sh -o ./results -v

# 使用自定义基线
./intra_node_bandwidth_check.sh -o ./results -b custom_baseline.json
```

**输出**：
```
结果目录/
├── gpu_info_<timestamp>.txt           # GPU信息
├── nvlink_topology_<timestamp>.txt    # NVLink拓扑
├── p2p_bandwidth_<timestamp>.txt      # P2P带宽测试原始输出
├── p2p_bandwidth_summary_<timestamp>.csv  # P2P带宽汇总
├── pcie_bandwidth_<timestamp>.txt     # PCIe带宽测试
├── pcie_bandwidth_summary_<timestamp>.csv # PCIe带宽汇总
├── slow_connections_<timestamp>.txt   # 慢连接列表（如有）
└── bandwidth_check_report_<timestamp>.md  # 综合报告
```

### 2. inter_node_nccl_check.sh

**功能**：检测多节点之间的NCCL通讯性能

**依赖**：
- MPI (OpenMPI or MPICH)
- nccl-tests (https://github.com/NVIDIA/nccl-tests)
- SSH免密登录到所有节点

**用法**：
```bash
# 创建节点列表文件
cat > nodes.txt <<EOF
gpu-node1
gpu-node2
gpu-node3
gpu-node4
EOF

# 基本用法
./inter_node_nccl_check.sh -n nodes.txt -o ./results

# 启用成对测试
./inter_node_nccl_check.sh -n nodes.txt -o ./results --pairwise

# 启用二分搜索
./inter_node_nccl_check.sh -n nodes.txt -o ./results --binary-search

# 自定义迭代次数和消息大小
./inter_node_nccl_check.sh -n nodes.txt -o ./results \
  -i 20 -s 16G

# 指定MPI和NCCL测试路径
./inter_node_nccl_check.sh -n nodes.txt -o ./results \
  --mpi-path /opt/openmpi \
  --nccl-tests-path /opt/nccl-tests/build
```

**输出**：
```
结果目录/
├── all_nodes_<timestamp>_iter1.txt    # 全节点测试迭代1
├── all_nodes_<timestamp>_iter2.txt    # 全节点测试迭代2
├── ...
├── all_nodes_<timestamp>_stats.txt    # 全节点统计
├── pairwise_results_<timestamp>.csv   # 成对测试结果
├── binary_search_*.txt                # 二分搜索结果（如启用）
└── nccl_check_report_<timestamp>.md   # 综合报告
```

### 3. detect_slow_nodes.sh

**功能**：综合检测工具，整合节点内部和跨节点检测

**用法**：
```bash
# 完整集群检查
./detect_slow_nodes.sh -n nodes.txt -o ./results

# 仅节点内部检查
./detect_slow_nodes.sh -n nodes.txt --skip-inter

# 仅跨节点检查
./detect_slow_nodes.sh -n nodes.txt --skip-intra

# 并行运行节点内部检查 + 成对测试 + 二分搜索
./detect_slow_nodes.sh -n nodes.txt -o ./results \
  --parallel --pairwise --binary-search

# 自定义阈值和迭代次数
./detect_slow_nodes.sh -n nodes.txt -o ./results \
  -t 85 --nccl-iterations 20
```

**输出**：
```
结果目录/
├── intra_node_results/          # 所有节点的内部检查结果
│   ├── node1_<timestamp>/
│   ├── node2_<timestamp>/
│   └── ...
├── inter_node_results/          # 跨节点NCCL测试结果
│   ├── all_nodes_*.txt
│   ├── pairwise_results_*.csv
│   └── nccl_check_report_*.md
└── slow_node_summary_<timestamp>.md  # 综合汇总报告
```

### 4. Ansible角色：slow_node_detection

**功能**：自动化在整个集群运行慢节点检测

**用法**：

1. **创建inventory文件**：
```ini
# inventory.ini
[gpu_cluster]
gpu-node1 ansible_host=192.168.1.10
gpu-node2 ansible_host=192.168.1.11
gpu-node3 ansible_host=192.168.1.12
gpu-node4 ansible_host=192.168.1.13
```

2. **运行playbook**：
```bash
# 基本用法
ansible-playbook -i inventory.ini playbooks/detect_slow_nodes.yml

# 自定义配置
ansible-playbook -i inventory.ini playbooks/detect_slow_nodes.yml \
  -e slow_node_detection_threshold=85 \
  -e slow_node_detection_pairwise=true \
  -e slow_node_detection_binary_search=true

# 仅节点内部检查（快速）
ansible-playbook -i inventory.ini playbooks/detect_slow_nodes.yml \
  -e slow_node_detection_skip_inter=true

# 并行模式（更快）
ansible-playbook -i inventory.ini playbooks/detect_slow_nodes.yml \
  -e slow_node_detection_parallel=true
```

3. **角色变量**（在playbook中或命令行指定）：
```yaml
# 结果输出目录
slow_node_detection_output_dir: "/tmp/slow_node_detection"

# 性能阈值（百分比）
slow_node_detection_threshold: 90

# NCCL测试配置
slow_node_detection_nccl_iterations: 10
slow_node_detection_nccl_message_size: "8G"

# 测试选项
slow_node_detection_skip_intra: false
slow_node_detection_skip_inter: false
slow_node_detection_pairwise: true
slow_node_detection_binary_search: true
slow_node_detection_parallel: true
```

---

## 性能基线

### GPU NVLink带宽基线

#### NVIDIA A100 (80GB SXM4)

**配置**：8个GPU，每个GPU有12条NVLink 3.0链路

| GPU对 | 连接方式 | 预期带宽 | 可接受阈值(90%) |
|-------|---------|----------|----------------|
| 任意两个GPU | NVLink 3.0 | 300-400 GB/s | 270 GB/s |

**nvidia-smi拓扑示例**：
```
GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7
GPU0     X      NV12    NV12    NV12    NV12    NV12    NV12    NV12
GPU1    NV12     X      NV12    NV12    NV12    NV12    NV12    NV12
...
```

#### NVIDIA H100 (80GB SXM5)

**配置**：8个GPU，每个GPU有18条NVLink 4.0链路

| GPU对 | 连接方式 | 预期带宽 | 可接受阈值(90%) |
|-------|---------|----------|----------------|
| 任意两个GPU | NVLink 4.0 | 450-600 GB/s | 405 GB/s |

#### NVIDIA V100 (32GB SXM2)

**配置**：8个GPU，每个GPU有6条NVLink 2.0链路

| GPU对 | 连接方式 | 预期带宽 | 可接受阈值(90%) |
|-------|---------|----------|----------------|
| 任意两个GPU | NVLink 2.0 | 150-200 GB/s | 135 GB/s |

### PCIe带宽基线

| GPU型号 | PCIe版本 | HtoD带宽 | DtoH带宽 | 可接受阈值(90%) |
|---------|---------|----------|----------|----------------|
| A100 PCIe | Gen4 x16 | ~26 GB/s | ~28 GB/s | 23 GB/s |
| H100 PCIe | Gen5 x16 | ~52 GB/s | ~56 GB/s | 47 GB/s |
| V100 PCIe | Gen3 x16 | ~14 GB/s | ~16 GB/s | 12.6 GB/s |

### NCCL All-Reduce带宽基线

#### 节点内（8 GPUs）

| GPU型号 | 网络 | Bus Bandwidth | 可接受阈值(92%) |
|---------|------|---------------|----------------|
| A100 SXM4 | NVLink 3.0 | ~250 GB/s | 230 GB/s |
| H100 SXM5 | NVLink 4.0 | ~350 GB/s | 322 GB/s |
| V100 SXM2 | NVLink 2.0 | ~180 GB/s | 165 GB/s |

#### 跨节点（InfiniBand）

| GPU型号 | IB版本 | Bus Bandwidth | 可接受阈值(92%) |
|---------|--------|---------------|----------------|
| A100 | HDR 200Gb | ~180 GB/s | 165 GB/s |
| A100 | NDR 400Gb | ~360 GB/s | 331 GB/s |
| H100 | NDR 400Gb | ~360 GB/s | 331 GB/s |

#### 跨节点（RoCE）

| GPU型号 | RoCE速度 | Bus Bandwidth | 可接受阈值(92%) |
|---------|---------|---------------|----------------|
| A100 | 100GbE | ~90 GB/s | 82 GB/s |
| A100 | 200GbE | ~180 GB/s | 165 GB/s |
| H100 | 400GbE | ~360 GB/s | 331 GB/s |

---

## 故障排查

### 问题1：NVLink带宽低于预期

**症状**：
```
GPU 0 <-> GPU 1: 150 GB/s (预期: 300 GB/s, A100 SXM4)
```

**可能原因及解决方法**：

1. **NVLink链路inactive**
   ```bash
   # 检查
   nvidia-smi nvlink --status -i 0

   # 如发现inactive链路
   Link 0: inactive
   Link 1: inactive
   ```

   **解决方法**：
   - 检查NVLink cable连接
   - 重启节点
   - 更换故障cable
   - 联系硬件厂商

2. **NVLink运行在降速模式**
   ```bash
   # 检查NVLink版本和速度
   nvidia-smi nvlink --capabilities -i 0
   ```

   **解决方法**：
   - 更新GPU固件
   - 更新NVIDIA驱动
   - 检查BIOS设置

3. **GPU温度过高导致降频**
   ```bash
   # 检查温度
   nvidia-smi --query-gpu=temperature.gpu --format=csv
   ```

   **解决方法**：
   - 检查散热系统
   - 清理灰尘
   - 降低环境温度

### 问题2：PCIe带宽低于预期

**症状**：
```
GPU 0: Gen3 x8, HtoD: 8 GB/s (预期: Gen4 x16, 26 GB/s)
```

**可能原因及解决方法**：

1. **PCIe槽位配置错误**
   ```bash
   # 检查当前PCIe配置
   nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv

   # 检查最大支持配置
   nvidia-smi --query-gpu=pcie.link.gen.max,pcie.link.width.max --format=csv
   ```

   **解决方法**：
   - 确认GPU安装在正确的PCIe槽位（通常是最靠近CPU的x16槽位）
   - 检查主板文档，确认槽位支持的PCIe代数
   - 更新主板BIOS

2. **PCIe ACS导致降速**
   ```bash
   # 检查IOMMU/ACS状态
   dmesg | grep -i iommu
   ```

   **解决方法**：
   - 在BIOS中禁用ACS（如果不需要IOMMU隔离）
   - 或在内核参数中添加 `pcie_aspm=off`

3. **PCIe错误累积**
   ```bash
   # 检查PCIe错误
   nvidia-smi --query-gpu=pci.replay_counter --format=csv
   ```

   **解决方法**：
   - 重启系统清除错误
   - 检查PCIe插槽是否有异物
   - 重新插拔GPU卡

### 问题3：跨节点NCCL性能差

**症状**：
```
All-reduce bus bandwidth: 80 GB/s (预期: 180 GB/s, IB HDR)
```

**可能原因及解决方法**：

1. **InfiniBand链路问题**
   ```bash
   # 检查IB状态（在所有节点）
   ibstat

   # 检查IB链路速度和宽度
   ibstatus

   # 预期输出示例：
   # State: Active
   # Rate: 200 (HDR)
   # Link layer: InfiniBand
   ```

   **解决方法**：
   - 确认IB链路为Active状态
   - 检查IB cable连接
   - 重启RDMA服务：`systemctl restart rdma`
   - 检查交换机端口状态

2. **GPU Direct RDMA未启用**
   ```bash
   # 检查nv_peer_mem模块
   lsmod | grep nv_peer_mem

   # 检查NCCL配置
   export NCCL_DEBUG=INFO
   # 运行NCCL测试，查看是否使用GPU Direct RDMA
   ```

   **解决方法**：
   - 安装nv_peer_mem模块
   - 确保NCCL版本支持GPU Direct RDMA
   - 设置 `NCCL_IB_GID_INDEX=3` (RoCE) 或 `0` (IB)

3. **网络拥塞**
   ```bash
   # 检查网络流量
   iftop -i ib0

   # 检查NCCL环境变量
   export NCCL_DEBUG=INFO
   export NCCL_NET_GDR_LEVEL=5
   ```

   **解决方法**：
   - 隔离GPU流量到专用网络
   - 调整NCCL拓扑：`export NCCL_TOPO_FILE=/path/to/topology.xml`
   - 使用QoS限制非关键流量

4. **成对测试发现特定节点慢**
   ```
   node1 <-> node3: 80 GB/s ✗
   node2 <-> node3: 85 GB/s ✗
   node3 <-> node4: 78 GB/s ✗
   其他节点对: 180 GB/s ✓

   分析：node3是问题节点
   ```

   **解决方法**：
   - 重点检查node3的IB连接
   - 检查node3的RDMA驱动和固件
   - 检查交换机端口（连接node3的端口）
   - 替换node3的IB cable
   - 如问题持续，考虑更换node3的IB卡

### 问题4：性能不稳定（标准差大）

**症状**：
```
迭代1: 240 GB/s
迭代2: 180 GB/s
迭代3: 235 GB/s
迭代4: 175 GB/s
...
标准差: 28 GB/s (>10%)
```

**可能原因及解决方法**：

1. **GPU频率波动（温度throttling）**
   ```bash
   # 监控GPU频率和温度
   watch -n 1 nvidia-smi dmon -s pucmet
   ```

   **解决方法**：
   - 改善散热
   - 设置GPU persistence mode：`nvidia-smi -pm 1`
   - 锁定GPU频率：`nvidia-smi -lgc <min>,<max>`

2. **后台任务干扰**
   ```bash
   # 检查GPU使用情况
   nvidia-smi pmon -c 10

   # 检查CPU使用情况
   top
   ```

   **解决方法**：
   - 停止非必要的后台任务
   - 使用cgroups隔离资源
   - 使用SLURM或Kubernetes独占节点

3. **网络间歇性问题**
   ```bash
   # 检查网络错误
   ethtool -S ib0 | grep -i error

   # 检查IB错误计数器
   perfquery
   ```

   **解决方法**：
   - 检查并更换有问题的网络cable
   - 检查交换机日志
   - 调整网络MTU和其他参数

---

## 最佳实践

### 1. 定期检测

**建议频率**：
- **新集群部署后**：立即运行完整检测
- **日常运行**：每周运行快速检测（仅节点内部）
- **定期维护**：每月运行完整检测（包括跨节点）
- **问题排查**：发现性能问题时立即运行
- **硬件更换后**：立即运行相关检测

**自动化检测**：
```bash
# 使用cron定期运行
# /etc/cron.weekly/gpu_cluster_check
#!/bin/bash
cd /opt/gpu_passthrough/ansible
ansible-playbook -i production_inventory playbooks/detect_slow_nodes.yml \
  -e slow_node_detection_skip_inter=true \
  -e slow_node_detection_output_dir=/var/log/slow_node_detection/$(date +%Y%m%d)
```

### 2. 建立性能基线

在集群正常运行时建立基线：

```bash
# 1. 运行检测并保存结果作为基线
./detect_slow_nodes.sh -n nodes.txt -o baseline_results \
  --pairwise --nccl-iterations 20

# 2. 提取基线数据
baseline_busbw=$(grep "Mean:" baseline_results/all_nodes_*_stats.txt | awk '{print $2}')
echo "Baseline Bus BW: $baseline_busbw GB/s" > cluster_baseline.txt

# 3. 在后续检测中使用基线
./detect_slow_nodes.sh -n nodes.txt -o daily_check \
  -b cluster_baseline.txt
```

### 3. 分层检测策略

**第一层：快速检测（5-10分钟）**
```bash
# 仅节点内部检查，无NCCL测试
./detect_slow_nodes.sh -n nodes.txt --skip-inter --parallel
```

**第二层：标准检测（30-60分钟）**
```bash
# 节点内部 + 全节点NCCL测试
./detect_slow_nodes.sh -n nodes.txt
```

**第三层：深度检测（2-4小时）**
```bash
# 完整检测：节点内部 + 成对测试 + 二分搜索
./detect_slow_nodes.sh -n nodes.txt \
  --pairwise --binary-search \
  --nccl-iterations 20 --parallel
```

### 4. 结果分析和记录

**建立检测日志**：
```bash
# 创建集中化日志目录
mkdir -p /var/log/slow_node_detection/{daily,weekly,monthly}

# 保存检测结果
cp -r results /var/log/slow_node_detection/daily/$(date +%Y%m%d)

# 创建趋势分析
cat > /var/log/slow_node_detection/trend_analysis.sh <<'EOF'
#!/bin/bash
# 分析最近30天的检测结果
for dir in /var/log/slow_node_detection/daily/*/; do
  date=$(basename $dir)
  busbw=$(grep "Mean:" $dir/all_nodes_*_stats.txt | awk '{print $2}')
  echo "$date,$busbw"
done | sort > bandwidth_trend.csv
EOF
```

### 5. 告警和通知

**集成监控系统**：
```bash
# 检测完成后发送告警
if grep -q "SLOW" results/pairwise_results_*.csv; then
  # 发送告警到Slack
  curl -X POST -H 'Content-type: application/json' \
    --data '{"text":"⚠️ Slow nodes detected in GPU cluster!"}' \
    $SLACK_WEBHOOK_URL

  # 发送邮件
  mail -s "GPU Cluster Slow Node Alert" admin@example.com < results/summary.txt
fi
```

### 6. 问题隔离流程

发现慢节点后的标准流程：

```
1. 隔离问题节点
   └─> 从调度系统移除：slurm down node3

2. 详细检查
   └─> 运行单节点深度检测
   └─> 检查硬件日志
   └─> 检查系统日志

3. 尝试修复
   └─> 重启节点
   └─> 更新驱动/固件
   └─> 重新插拔硬件

4. 验证修复
   └─> 重新运行检测
   └─> 与基线对比
   └─> 成对测试验证

5. 恢复服务或替换硬件
   └─> 如验证通过：恢复到调度系统
   └─> 如验证失败：标记硬件故障，申请更换
```

### 7. 文档化和知识库

**维护问题知识库**：
```markdown
# 故障案例库

## 案例1：node5 NVLink带宽低
- **日期**：2024-12-15
- **症状**：GPU 2<->GPU 3带宽仅120 GB/s（预期300 GB/s）
- **根因**：NVLink cable松动
- **解决**：重新插拔cable，验证通过
- **预防**：定期检查cable连接

## 案例2：node8跨节点通讯慢
- **日期**：2024-12-20
- **症状**：所有与node8的成对测试均<100 GB/s
- **根因**：InfiniBand卡固件版本过旧
- **解决**：升级IB卡固件到最新版本
- **预防**：建立固件版本管理流程
```

---

## 案例分析

### 案例1：新集群部署验收

**场景**：
- 新采购的4节点GPU集群（每节点8个A100 80GB SXM4）
- InfiniBand HDR 200Gb网络
- 需要验收并确保性能达标

**检测流程**：

1. **运行完整检测**：
```bash
./detect_slow_nodes.sh -n nodes.txt -o acceptance_test \
  --pairwise --binary-search --parallel \
  --nccl-iterations 20 -t 92
```

2. **检测结果**：
```
节点内部检测：
  ✓ 所有节点NVLink带宽正常（>270 GB/s）
  ✓ 所有节点PCIe Gen4 x16
  ✓ 无慢GPU连接

跨节点检测：
  ✓ 全节点all-reduce: 178 GB/s (预期180 GB/s, 阈值165 GB/s)
  ⚠ 成对测试发现问题：
      node1 <-> node3: 95 GB/s ✗
      node2 <-> node3: 92 GB/s ✗
      node3 <-> node4: 98 GB/s ✗
      其他节点对: 175-180 GB/s ✓

  分析：node3在所有成对测试中均慢 → node3有问题
```

3. **问题排查**：
```bash
# 在node3上检查IB状态
ssh node3 ibstat

# 发现问题
Port 1:
    State: Active
    Rate: 100  # ✗ 应该是200 (HDR)
```

4. **解决方案**：
- 检查IB cable，发现使用了旧的EDR cable（100Gb）而非HDR cable（200Gb）
- 更换为HDR cable
- 重新运行检测：
```
成对测试：
  node1 <-> node3: 177 GB/s ✓
  node2 <-> node3: 179 GB/s ✓
  node3 <-> node4: 178 GB/s ✓
```

5. **结论**：
- 集群通过验收
- 文档记录：确保所有IB连接使用HDR cable

### 案例2：生产集群性能下降

**场景**：
- 16节点生产集群运行6个月后
- 用户反馈训练速度变慢
- 需要定位性能下降原因

**检测流程**：

1. **对比历史基线**：
```bash
# 运行当前检测
./detect_slow_nodes.sh -n nodes.txt -o current_check

# 对比6个月前的基线
# 基线：250 GB/s
# 当前：195 GB/s (-22%)
```

2. **二分搜索定位**：
```bash
# 启用二分搜索
./inter_node_nccl_check.sh -n nodes.txt --binary-search

# 结果
全节点(1-16): 195 GB/s ✗
组1(1-8): 248 GB/s ✓
组2(9-16): 180 GB/s ✗
组2-1(9-12): 245 GB/s ✓
组2-2(13-16): 175 GB/s ✗
组2-2-1(13-14): 242 GB/s ✓
组2-2-2(15-16): 125 GB/s ✗
节点15: 240 GB/s ✓
节点16: ✗ 问题节点
```

3. **节点16详细检查**：
```bash
# 节点内部检测
ssh node16 "/opt/scripts/intra_node_bandwidth_check.sh -o /tmp/node16_check"

# 发现问题
GPU 0 <-> GPU 3: 85 GB/s ✗ (预期300 GB/s)
GPU 1 <-> GPU 3: 88 GB/s ✗
GPU 2 <-> GPU 3: 280 GB/s ✓
GPU 3 <-> GPU 4: 90 GB/s ✗
GPU 3 <-> GPU 5: 87 GB/s ✗
GPU 3 <-> GPU 6: 92 GB/s ✗
GPU 3 <-> GPU 7: 85 GB/s ✗

# 检查GPU 3 NVLink状态
nvidia-smi nvlink --status -i 3

# 发现多个NVLink链路inactive
Link 0: inactive
Link 1: inactive
Link 2: inactive
Link 4: inactive
```

4. **解决方案**：
- GPU 3的NVLink链路故障
- 由于在质保期内，联系厂商更换GPU 3
- 临时方案：使用CUDA_VISIBLE_DEVICES隔离GPU 3，节点以7个GPU运行

5. **验证修复**：
```bash
# 更换GPU后重新检测
./detect_slow_nodes.sh -n nodes.txt -o post_fix_check

# 结果
全节点all-reduce: 248 GB/s ✓ (恢复到基线)
所有成对测试: >175 GB/s ✓
node16所有GPU间带宽: >270 GB/s ✓
```

6. **经验总结**：
- 定期运行检测可及早发现硬件衰减
- 二分搜索可快速定位问题节点
- 节点内部检测可精确定位故障GPU
- 建立备件库和快速响应流程

### 案例3：多节点间歇性慢节点

**场景**：
- 32节点集群
- 训练任务偶尔出现性能波动
- 难以复现和定位

**检测策略**：

1. **增加检测频率和迭代次数**：
```bash
# 每小时运行一次，每次50次迭代
while true; do
  ./inter_node_nccl_check.sh -n nodes.txt \
    -o "check_$(date +%Y%m%d_%H%M%S)" \
    -i 50 --pairwise
  sleep 3600
done
```

2. **分析统计数据**：
```bash
# 收集所有检测的统计数据
for dir in check_*/; do
  grep "Mean\|StdDev" $dir/all_nodes_*_stats.txt
done > all_stats.txt

# 分析标准差
awk '/StdDev/ {print $2}' all_stats.txt | sort -n

# 发现某些时段标准差显著增大(>15 GB/s)
```

3. **识别模式**：
```bash
# 标记高标准差时段
02:15 - StdDev: 18.5 GB/s
02:20 - StdDev: 22.3 GB/s
10:15 - StdDev: 19.8 GB/s
10:20 - StdDev: 21.5 GB/s
...

# 发现模式：02:00-03:00 和 10:00-11:00 时段
```

4. **关联分析**：
```bash
# 检查这些时段的系统事件
grep "02:00\|10:00" /var/log/syslog

# 发现
02:00 - 每日备份任务启动（占用网络带宽）
10:00 - 集群监控数据同步（NCCL测试与监控流量冲突）
```

5. **解决方案**：
- 将备份任务移到专用管理网络
- 调整监控数据收集策略，避免与训练时段重叠
- 使用QoS隔离训练流量和管理流量

6. **验证**：
```bash
# 调整后连续监控7天
# 标准差始终<5 GB/s ✓
# 用户反馈性能稳定 ✓
```

---

## 总结

### 工具选择指南

| 场景 | 推荐工具 | 预计时间 |
|------|---------|---------|
| 单节点快速检查 | intra_node_bandwidth_check.sh | 5分钟 |
| 小集群(<4节点)完整检测 | detect_slow_nodes.sh | 30分钟 |
| 中型集群(4-16节点) | detect_slow_nodes.sh --pairwise | 1-2小时 |
| 大型集群(16+节点) | detect_slow_nodes.sh --binary-search | 2-4小时 |
| 自动化定期检测 | Ansible playbook | 按规模 |

### 性能优化建议

1. **并行执行**：使用 `--parallel` 加速节点内部检查
2. **智能策略**：
   - 日常：仅运行节点内部检查
   - 周期：运行全节点all-reduce测试
   - 问题排查：启用成对测试和二分搜索
3. **增量检测**：只检测新加入或报告问题的节点

### 集成建议

1. **CI/CD集成**：在集群部署pipeline中加入自动化检测
2. **监控系统集成**：将检测结果推送到Prometheus/Grafana
3. **工单系统集成**：自动创建硬件故障工单
4. **文档系统集成**：自动更新集群健康状况文档

### 进一步资源

- **NVIDIA NCCL文档**：https://docs.nvidia.com/deeplearning/nccl/
- **NCCL Tests GitHub**：https://github.com/NVIDIA/nccl-tests
- **Together.AI GPU集群测试指南**：https://www.together.ai/blog/a-practitioners-guide-to-testing-and-running-large-gpu-clusters
- **NVIDIA Multi-Node调优指南**：https://docs.nvidia.com/multi-node-nvlink-systems/

---

**文档版本**：1.0
**最后更新**：2025-01-17

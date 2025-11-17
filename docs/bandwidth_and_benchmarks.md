# GPU 通讯带宽测试和模型训练基准测试指南

## 概述

本文档介绍如何测试和验证 GPU 服务器的通讯带宽以及实际模型训练性能，并与性能基线对比。

## 测试覆盖范围

### 1. **机内通讯带宽**
- ✅ **PCIe 带宽**: CPU-GPU 之间的数据传输
- ✅ **NVLink 带宽**: GPU-GPU 之间的高速互连 (如可用)
- ✅ **GPU 内存带宽**: 设备内存访问性能

### 2. **机间通讯带宽**
- ✅ **RDMA 带宽**: InfiniBand/RoCE 网络性能
- ✅ **GPUDirect RDMA**: GPU 直接访问远程 GPU 内存

### 3. **实际模型训练**
- ✅ **NCCL 集合通信**: AllReduce, Broadcast 等操作
- ✅ **Megatron-LM 训练**: GPT 模型实际训练吞吐量
- ✅ **分布式训练性能**: 多GPU/多节点扩展性

## 一、性能基线数据库

### GPU 性能基线

我们提供了主流 GPU 型号的性能基线数据：

| GPU 型号 | FP16 TFLOPS | 内存带宽 | NVLink BW | PCIe Gen | 预期 MFU |
|---------|-------------|----------|-----------|----------|---------|
| A100-SXM4-80GB | 312 | 2039 GB/s | 600 GB/s | Gen4 | 52% |
| H100-SXM5-80GB | 756 | 3350 GB/s | 900 GB/s | Gen5 | 47% |
| V100-SXM2-32GB | 125 | 900 GB/s | 300 GB/s | Gen3 | 30% |

### NCCL 预期性能

| GPU 型号 | 节点内 AllReduce | 节点间 (IB HDR) |
|---------|-----------------|----------------|
| A100-SXM4 (8GPU) | 250 GB/s | 180 GB/s |
| H100-SXM5 (8GPU) | 450 GB/s | 350 GB/s |
| V100-SXM2 (8GPU) | 180 GB/s | 90 GB/s |

### 查看性能基线

```bash
# 列出所有可用的 GPU 型号
python3 /opt/gpu-benchmarks/performance_baselines.py list

# 查看特定 GPU 的详细信息
python3 /opt/gpu-benchmarks/performance_baselines.py info A100-SXM4-80GB

# 对比两个 GPU 型号
python3 /opt/gpu-benchmarks/performance_baselines.py compare H100-SXM5-80GB A100-SXM4-80GB

# 导出基线数据为 JSON
python3 /opt/gpu-benchmarks/performance_baselines.py export baselines.json
```

## 二、带宽测试

### 2.1 安装测试工具

```bash
# 使用 Ansible 安装所有测试工具
cd ansible
ansible-playbook playbooks/install_benchmark_tools.yml

# 或使用 benchmark_tools role
ansible-playbook -i inventory/hosts.yml playbooks/install_benchmark_tools.yml
```

安装的工具包括：
- CUDA samples (bandwidthTest, p2pBandwidthLatencyTest)
- nvbandwidth (NVIDIA 官方带宽测试工具)
- NCCL tests
- perftest (RDMA/InfiniBand 测试)
- Megatron-LM (可选)

### 2.2 PCIe 带宽测试

#### 方法 1: 使用 bandwidthTest (CUDA Samples)

```bash
# 基本测试
bandwidthTest

# 详细测试所有传输模式
bandwidthTest --htod --dtoh --dtod

# 指定设备
bandwidthTest --device=0
```

**预期结果**:
- **PCIe Gen3 x16**: ~12 GB/s (单向)
- **PCIe Gen4 x16**: ~24 GB/s (单向)
- **PCIe Gen5 x16**: ~48 GB/s (单向)

#### 方法 2: 使用 nvbandwidth

```bash
# Host-to-Device 和 Device-to-Host 测试
nvbandwidth -t host_to_device_memcpy_ce,device_to_host_memcpy_ce

# 所有内存拷贝模式
nvbandwidth -t all
```

#### 方法 3: 使用综合测试脚本

```bash
# 运行完整的带宽测试套件
/opt/gpu-benchmarks/bandwidth_test.sh

# 或使用包装器
gpu-benchmark bandwidth

# 指定输出目录
gpu-benchmark bandwidth /tmp/my_bandwidth_tests
```

测试脚本会自动：
1. 检测 GPU 型号和数量
2. 运行 PCIe 带宽测试
3. 检查 PCIe 链路状态
4. 与性能基线对比
5. 生成 JSON 报告

### 2.3 NVLink 带宽测试

#### 检查 NVLink 拓扑

```bash
# 查看 GPU 拓扑和 NVLink 连接
nvidia-smi topo -m

# 查看 NVLink 状态
nvidia-smi nvlink --status
```

#### P2P 带宽测试

```bash
# GPU 间点对点带宽测试
p2pBandwidthLatencyTest

# 使用 nvbandwidth 测试设备间传输
nvbandwidth -t device_to_device_memcpy_read_ce,device_to_device_memcpy_write_ce
```

**预期结果**:
- **NVLink 2.0 (V100)**: ~35 GB/s (单向), ~280 GB/s (全双工)
- **NVLink 3.0 (A100)**: ~50 GB/s (单向), ~430-520 GB/s (双向 8GPU)
- **NVLink 4.0 (H100)**: ~75 GB/s (单向), ~900 GB/s (双向 8GPU)

### 2.4 RDMA 带宽测试

#### 检查 InfiniBand/RoCE

```bash
# 查看 IB 设备状态
ibstat

# 查看 IB 设备列表
ibstat -l

# 检查链路速度
ibstat | grep -E "State|Rate|Link"
```

#### RDMA 带宽测试 (需要两个节点)

**节点 1 (服务器)**:
```bash
# 获取 IB 设备名
IB_DEV=$(ibstat -l | head -1)

# 启动服务器
ib_write_bw -d $IB_DEV
```

**节点 2 (客户端)**:
```bash
# 运行客户端测试
ib_write_bw -d $IB_DEV <server_ip_address>
```

**预期结果**:
- **InfiniBand EDR**: ~11.5 GB/s (100 Gbps)
- **InfiniBand HDR**: ~23 GB/s (200 Gbps)
- **InfiniBand NDR**: ~46 GB/s (400 Gbps)
- **RoCE v2 100G**: ~11 GB/s
- **RoCE v2 200G**: ~22 GB/s

#### GPUDirect RDMA 测试

如果 perftest 编译时启用了 CUDA 支持：

**服务器**:
```bash
ib_write_bw -d $IB_DEV --use_cuda=0
```

**客户端**:
```bash
ib_write_bw -d $IB_DEV --use_cuda=0 <server_ip>
```

## 三、NCCL 集合通信测试

### 3.1 安装 NCCL Tests

```bash
# 通过 Ansible 自动安装
ansible-playbook playbooks/install_benchmark_tools.yml -t nccl_tests

# 或手动安装
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make MPI=1
```

### 3.2 单节点 NCCL 测试

```bash
# 使用测试脚本（推荐）
/opt/gpu-benchmarks/nccl_benchmark.sh

# 或使用包装器
gpu-benchmark nccl

# 直接运行 NCCL tests
export NCCL_TESTS_DIR=/opt/nccl-tests/build

# AllReduce 测试 (8 GPUs)
$NCCL_TESTS_DIR/all_reduce_perf -b 8 -e 8G -f 2 -g 8

# Broadcast 测试
$NCCL_TESTS_DIR/broadcast_perf -b 8 -e 8G -f 2 -g 8

# Reduce-Scatter 测试
$NCCL_TESTS_DIR/reduce_scatter_perf -b 8 -e 8G -f 2 -g 8

# All-Gather 测试
$NCCL_TESTS_DIR/allgather_perf -b 8 -e 8G -f 2 -g 8
```

### 3.3 多节点 NCCL 测试

```bash
# 使用 MPI 运行多节点测试
# 64 GPUs (8 nodes x 8 GPUs)
mpirun -np 64 -N 8 \
    --hostfile hostfile \
    --bind-to none \
    --allow-run-as-root \
    $NCCL_TESTS_DIR/all_reduce_perf -b 8 -e 8G -f 2 -g 1
```

**hostfile 示例**:
```
node1 slots=8
node2 slots=8
node3 slots=8
node4 slots=8
```

### 3.4 理解 NCCL 输出

NCCL tests 输出两个关键指标：

1. **Algorithm Bandwidth (algbw)**: 算法带宽，从应用角度看到的有效带宽
2. **Bus Bandwidth (busbw)**: 总线带宽，物理总线上的实际数据传输量

**示例输出**:
```
#       size         count      type   redop    root     time   algbw   busbw
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)
     8388608       2097152     float     sum      -1   4521.7   185.6   348.5
```

**性能判断**:
- **busbw** 应接近或达到基线值
- **A100 8GPU 节点内**: 期望 busbw ~250 GB/s
- **H100 8GPU 节点内**: 期望 busbw ~450 GB/s
- **跨节点 (IB HDR)**: 期望 busbw ~180 GB/s

## 四、Megatron-LM 模型训练基准

### 4.1 安装 Megatron-LM

```bash
# 通过 Ansible 安装
ansible-playbook playbooks/install_benchmark_tools.yml -e "install_megatron=true"

# 或手动安装
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
pip install -r requirements.txt
```

### 4.2 运行训练基准

```bash
# 使用基准测试脚本
/opt/gpu-benchmarks/megatron_benchmark.sh

# 或使用包装器
gpu-benchmark megatron

# 指定模型大小
MODEL_SIZE=GPT-1.2B gpu-benchmark megatron

# 自定义参数
MODEL_SIZE=GPT-8.3B BATCH_SIZE=16 NUM_STEPS=200 \
    /opt/gpu-benchmarks/megatron_benchmark.sh
```

### 4.3 实际 Megatron 训练示例

#### GPT-1.2B 训练 (8x A100)

```bash
export MEGATRON_DIR=/opt/Megatron-LM

# 单节点 8 GPU 训练
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port=6000 \
    $MEGATRON_DIR/pretrain_gpt.py \
    --num-layers 24 \
    --hidden-size 2048 \
    --num-attention-heads 32 \
    --micro-batch-size 8 \
    --global-batch-size 64 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --train-iters 100 \
    --data-path /path/to/data \
    --vocab-file /path/to/vocab.json \
    --merge-file /path/to/merges.txt \
    --lr 0.00015 \
    --lr-decay-style cosine \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --fp16 \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 2 \
    --log-interval 10
```

### 4.4 预期性能基线

#### GPT-1.2B (单 GPU)

| GPU 型号 | TFLOPS | MFU | Samples/sec |
|---------|--------|-----|-------------|
| V100 | 39 | 30% | 12 |
| A100 | 93.6 | 60% | 28 |
| H100 | 178 | 47% | 45 |

#### GPT-8.3B (512 GPUs)

| GPU 型号 | PFLOPS | 扩展效率 |
|---------|--------|---------|
| V100 | 15.1 | 76% |
| A100 | 35 | 85% |

#### GPT-175B (1024 GPUs)

| GPU 型号 | 训练时间 | PFLOPS |
|---------|---------|--------|
| A100 | 30 天 | 160 |
| H100 | 10 天 | 480 |

#### GPT-1T (3072 A100)

- **总算力**: 502 PFLOPS
- **单 GPU TFLOPS**: 163
- **MFU (模型利用率)**: 52%
- **扩展效率**: 98%

## 五、完整测试流程

### 5.1 快速验证流程 (30 分钟)

```bash
#!/bin/bash
# 快速 GPU 性能验证

echo "=== 1. 系统配置检查 ==="
/opt/gpu-benchmarks/bandwidth_test.sh /tmp/bandwidth_quick

echo "=== 2. NCCL 快速测试 ==="
export NCCL_TESTS_DIR=/opt/nccl-tests/build
$NCCL_TESTS_DIR/all_reduce_perf -b 128M -e 1G -f 2 -g $(nvidia-smi -L | wc -l)

echo "=== 3. 查看性能基线对比 ==="
GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -d ' ')
python3 /opt/gpu-benchmarks/performance_baselines.py info $GPU_MODEL

echo "=== 验证完成 ==="
```

### 5.2 完整验证流程 (2-4 小时)

```bash
#!/bin/bash
# 完整 GPU 性能验证和基准测试

OUTPUT_BASE=/tmp/gpu_full_validation_$(date +%Y%m%d)
mkdir -p $OUTPUT_BASE

echo "=== 1. 带宽测试 (30 分钟) ==="
/opt/gpu-benchmarks/bandwidth_test.sh $OUTPUT_BASE/bandwidth

echo "=== 2. NCCL 完整测试 (60 分钟) ==="
/opt/gpu-benchmarks/nccl_benchmark.sh $OUTPUT_BASE/nccl

echo "=== 3. Megatron 训练基准 (60-120 分钟) ==="
MODEL_SIZE=GPT-1.2B NUM_STEPS=200 \
    /opt/gpu-benchmarks/megatron_benchmark.sh $OUTPUT_BASE/megatron

echo "=== 4. 生成综合报告 ==="
python3 - <<EOF
import json
import glob

# 收集所有测试结果
results = {}

# 读取带宽测试结果
bw_files = glob.glob("$OUTPUT_BASE/bandwidth/*.json")
if bw_files:
    with open(bw_files[0]) as f:
        results['bandwidth'] = json.load(f)

# 读取 NCCL 测试结果
nccl_files = glob.glob("$OUTPUT_BASE/nccl/*summary*.json")
if nccl_files:
    with open(nccl_files[0]) as f:
        results['nccl'] = json.load(f)

# 读取 Megatron 测试结果
megatron_files = glob.glob("$OUTPUT_BASE/megatron/*summary*.json")
if megatron_files:
    with open(megatron_files[0]) as f:
        results['megatron'] = json.load(f)

# 保存综合报告
with open("$OUTPUT_BASE/final_report.json", 'w') as f:
    json.dump(results, f, indent=2)

print("\n综合测试报告已保存到: $OUTPUT_BASE/final_report.json")
EOF

echo "=== 验证完成 ==="
echo "结果目录: $OUTPUT_BASE"
```

## 六、性能问题诊断

### 6.1 PCIe 带宽低

**症状**: PCIe 带宽 < 预期的 50%

**诊断步骤**:
```bash
# 1. 检查 PCIe 链路状态
nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv

# 2. 检查是否运行在最大速度
nvidia-smi --query-gpu=pcie.link.gen.max,pcie.link.width.max --format=csv

# 3. 检查 IOMMU 设置
dmesg | grep -i iommu

# 4. 检查 BIOS PCIe 设置
# - PCIe Link Speed: Max (Gen4/Gen5)
# - PCIe ASPM: Disabled
# - Above 4G Decoding: Enabled
```

### 6.2 NVLink 带宽低

**症状**: P2P 带宽远低于预期

**诊断步骤**:
```bash
# 1. 检查 NVLink 状态
nvidia-smi nvlink --status

# 2. 查看拓扑
nvidia-smi topo -m

# 3. 检查 GPU 放置
lspci | grep NVIDIA

# 4. 验证 P2P 访问
nvidia-smi topo -p2p r
```

### 6.3 NCCL 性能低

**症状**: AllReduce busbw < 预期的 70%

**诊断步骤**:
```bash
# 1. 启用 NCCL 调试
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# 2. 检查 NCCL 使用的算法
export NCCL_ALGO=Ring  # 或 Tree

# 3. 检查网络设置（多节点）
export NCCL_IB_DISABLE=0  # 确保启用 IB
export NCCL_NET_GDR_LEVEL=5  # 启用 GPUDirect RDMA

# 4. 检查 GPU 亲和性
numactl --hardware
nvidia-smi topo -m

# 5. 重新运行测试
$NCCL_TESTS_DIR/all_reduce_perf -b 8 -e 8G -f 2 -g 8
```

### 6.4 训练吞吐量低

**症状**: 实际 TFLOPS < MFU * 峰值 TFLOPS * 70%

**诊断步骤**:
```bash
# 1. 检查 GPU 利用率
nvidia-smi dmon -s u

# 2. 检查是否有热节流
nvidia-smi -q -d TEMPERATURE,POWER

# 3. 检查数据加载瓶颈
# 使用 PyTorch Profiler 或 NVTX

# 4. 优化批大小和并行策略
# - 增加 micro-batch-size
# - 调整 tensor-parallel-size
# - 调整 pipeline-parallel-size
```

## 七、最佳实践

### 7.1 测试前准备

1. **环境配置**:
   ```bash
   # 设置 CPU 性能模式
   echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

   # 禁用 GPU ECC（测试时）
   sudo nvidia-smi -e 0

   # 设置 GPU 持久化模式
   sudo nvidia-smi -pm 1
   ```

2. **基线收集**:
   - 在干净环境下运行所有测试
   - 多次运行取平均值
   - 记录系统配置和环境变量

### 7.2 测试结果记录

建议记录以下信息：
- 完整系统配置 (CPU, GPU, 内存, 网络)
- BIOS 设置
- 内核参数
- CUDA/NCCL/MPI 版本
- 所有测试的原始输出
- 与基线的对比

### 7.3 性能优化建议

1. **PCIe 优化**:
   - 使用 PCIe Gen4/5
   - 禁用 ASPM
   - 启用 Resizable BAR

2. **NUMA 优化**:
   - 将 GPU 绑定到对应的 NUMA 节点
   - 使用 numactl 运行训练

3. **网络优化**:
   - 使用 GPUDirect RDMA
   - 优化 NCCL 参数
   - 使用高速网络 (IB NDR/HDR)

## 八、参考资源

### 官方文档
- [NVIDIA NCCL Tests](https://github.com/NVIDIA/nccl-tests)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [NCCL Performance Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/performance.html)
- [nvbandwidth](https://github.com/NVIDIA/nvbandwidth)

### 性能参考
- [A100 Deep Learning Benchmarks](https://lambda.ai/blog/nvidia-a100-gpu-deep-learning-benchmarks-and-architectural-overview)
- [H100 vs A100 Comparison](https://www.cudocompute.com/blog/comparative-analysis-of-nvidia-a100-vs-h100-gpus)
- [Scaling Megatron-LM](https://arxiv.org/pdf/1909.08053.pdf)

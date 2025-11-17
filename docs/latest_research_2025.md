# GPU 基线安装与验证 - 2024-2025 最新调研报告

## 更新时间: 2025-01-16

本文档包含最新的 GPU 部署工具、CPU 优化配置和系统验证最佳实践。

---

## 一、2024-2025 最新 GPU 部署工具

### 1.1 NVIDIA 官方工具更新

#### NVIDIA DeepOps (推荐用于裸金属集群)
- **项目地址**: https://github.com/NVIDIA/deepops
- **最新版本**: 22.04.1 (2022年4月发布，仍在维护)
- **描述**: NVIDIA 官方的 GPU 集群部署工具，封装了 GPU 服务器集群部署的最佳实践
- **核心特性**:
  - 支持 Kubernetes 和 Slurm 工作负载管理器
  - 裸金属和云环境支持
  - 使用 Ansible 和 Kubespray 作为底层
  - 包含 GPU 环境优化的额外 playbooks
  - 支持 DGX 系统
  - MAAS 裸金属自动化部署

**关键组件**:
```bash
# DeepOps 包含以下工具栈
- Ansible playbooks for system configuration
- Kubespray for Kubernetes deployment
- Slurm workload manager support
- GPU-optimized networking configurations
- Monitoring and logging stack
```

#### NVIDIA GPU Operator (推荐用于 Kubernetes)
- **更新**: 2024-2025 最活跃的项目
- **描述**: Kubernetes GPU 管理的标准化方案
- **核心功能**:
  - 自动化部署整个 GPU 软件栈
  - 支持 vGPU、MIG、Time Slicing
  - 支持 GPUDirect RDMA 和 Storage
  - 动态管理驱动和容器工具包

**关键特性** (2024-2025):
```yaml
Features:
  - Automated GPU stack deployment
  - Multi-Instance GPU (MIG) support
  - Time-slicing for GPU sharing
  - vGPU support
  - GPUDirect RDMA/Storage
  - Node Feature Discovery integration
```

#### NVIDIA Dynamo (2025 年新工具)
- **发布时间**: 2025 年初
- **项目**: NVIDIA Dynamo Kubernetes Operator
- **核心创新**:
  - AI 推理感知的自动扩展器
  - 单个 CLI 命令部署到数千 GPU
  - 动态资源管理（prefill 和 decode 阶段）
  - 监控工作负载模式并自动调整

**Dynamo v0.2 新功能**:
- NVIDIA Dynamo Planner: 推理感知的自动扩展
- 跨 prefill 和 decode 阶段的动态计算资源管理

#### NVIDIA KAI Scheduler (2025 年开源)
- **发布时间**: 2025 年 1 月
- **描述**: 企业级 GPU 管理调度器，专为 GPU 工作负载优化设计
- **特点**:
  - 专为 Kubernetes AI 工作负载设计
  - 高级 GPU 调度算法
  - 开源社区版本

### 1.2 社区最新项目

#### LIP-Computing/ansible-role-nvidia
- **更新**: 持续维护
- **特点**:
  - 支持可配置驱动版本
  - kmod_install 变量控制（Docker vs 裸金属）
  - 适合 VM 和裸金属环境

#### IBM MAS DevOps Collection
- **用途**: OpenShift 集群
- **包含**:
  - NVIDIA GPU Operator 安装
  - Node Feature Discovery (NFD) operator
  - 企业级 Kubernetes GPU 支持

---

## 二、CPU 性能优化配置 (GPU 工作负载)

基于 2024-2025 年最新研究和最佳实践。

### 2.1 NUMA (Non-Uniform Memory Access) 配置

#### 为什么 NUMA 重要？
- GPU 工作负载对内存访问延迟极为敏感
- 跨 NUMA 节点访问内存会导致显著性能下降
- 正确的 NUMA 配置可提升 20-40% 的性能

#### NUMA 最佳实践

**1. NUMA 模式选择**:
```bash
# AMD EPYC 平台
NPS=4  # 推荐用于内存带宽敏感的 MPI 应用
NPS=1  # 推荐用于未优化 NUMA 局部性的应用

# Intel 平台
Cluster-on-Die (CoD) / SNC (Sub-NUMA Clustering)
SNC4 类似于 AMD NPS4
```

**2. 检查 NUMA 配置**:
```bash
# 查看 NUMA 节点
numactl --hardware

# 查看 GPU 的 NUMA 亲和性
nvidia-smi topo -m

# 使用 rocm-smi 查看拓扑（AMD）
rocm-smi --showtopo

# 查看 PCIe 设备的 NUMA 节点
cat /sys/class/pci_bus/*/device/numa_node
```

**3. 绑定 CPU 和内存到 NUMA 节点**:
```bash
# PyTorch 示例：绑定到第 N 个 NUMA 节点
numactl --cpunodebind=N --membind=N python train.py

# 多 GPU 训练，每个 GPU 绑定到对应 NUMA 节点
# GPU 0 -> NUMA 0
numactl --cpunodebind=0 --membind=0 python -m torch.distributed.launch --nproc_per_node=1 --local_rank=0 train.py
```

**4. BIOS 设置**:
```
NUMA Mode: Enabled
NUMA Nodes per Socket (NPS): 4 (推荐用于 HPC/AI)
Memory Interleaving: Disabled (保持 NUMA 特性)
```

### 2.2 CPU 频率调节器 (CPU Governor)

#### 性能模式配置

**1. 设置为 Performance 模式**:
```bash
# 查看当前 governor
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# 设置所有 CPU 为 performance 模式
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance > $cpu
done

# 使用 cpupower 工具
cpupower frequency-set -g performance

# 永久设置（Ubuntu/Debian）
apt-get install cpufrequtils
echo 'GOVERNOR="performance"' > /etc/default/cpufrequtils
systemctl restart cpufrequtils
```

**2. 验证频率**:
```bash
# 查看当前频率
watch -n1 "grep MHz /proc/cpuinfo"

# 使用 cpupower
cpupower frequency-info
```

**3. 常见 Governor 模式**:
```
performance  - 静态最高频率（推荐用于 GPU 工作负载）
powersave    - 静态最低频率
ondemand     - 动态调整（根据负载）
conservative - 保守的动态调整
schedutil    - 基于调度器的动态调整
```

### 2.3 Turbo Boost / Turbo Core 配置

#### Intel Turbo Boost

**1. 启用 Turbo Boost**:
```bash
# 检查是否启用
cat /sys/devices/system/cpu/intel_pstate/no_turbo
# 0 = enabled, 1 = disabled

# 启用 Turbo Boost
echo 0 > /sys/devices/system/cpu/intel_pstate/no_turbo

# 检查 Turbo 频率
cpupower frequency-info
```

**2. BIOS 设置**:
```
Intel Turbo Boost Technology: Enabled
Processor Performance Boost Policy: Aggressive (Windows)
```

#### AMD Turbo Core / Precision Boost

**1. 启用 AMD Boost**:
```bash
# 检查 boost 状态
cat /sys/devices/system/cpu/cpufreq/boost
# 1 = enabled, 0 = disabled

# 启用 boost
echo 1 > /sys/devices/system/cpu/cpufreq/boost
```

**2. BIOS 设置**:
```
Core Performance Boost: Enabled
Precision Boost Overdrive: Enabled (Ryzen)
```

### 2.4 C-States 和 P-States 优化

#### 禁用深度 C-States（降低延迟）

**1. 内核参数**:
```bash
# 编辑 GRUB 配置
vim /etc/default/grub

# 添加到 GRUB_CMDLINE_LINUX_DEFAULT
GRUB_CMDLINE_LINUX_DEFAULT="intel_idle.max_cstate=1 processor.max_cstate=1"

# 或完全禁用 C-States
GRUB_CMDLINE_LINUX_DEFAULT="idle=poll"

# 更新 GRUB
update-grub
reboot
```

**2. BIOS 设置**:
```
C-States: Disabled 或 C1 only
Package C State: C0/C1 state
```

### 2.5 IOMMU 配置

#### 启用 IOMMU（必需用于 GPU 透传）

**1. Intel VT-d**:
```bash
# GRUB 配置
GRUB_CMDLINE_LINUX_DEFAULT="intel_iommu=on iommu=pt"

# pt (passthrough) 模式性能更好
```

**2. AMD IOMMU**:
```bash
GRUB_CMDLINE_LINUX_DEFAULT="amd_iommu=on iommu=pt"
```

**3. 验证 IOMMU**:
```bash
# 检查 IOMMU 组
find /sys/kernel/iommu_groups/ -type l

# 检查 IOMMU 设备
ls /sys/class/iommu/*/devices/

# dmesg 查看
dmesg | grep -i iommu
```

### 2.6 PCIe 优化

#### PCIe 性能设置

**1. BIOS 配置**:
```
PCIe Link Speed: Gen 4 / Gen 5 (最大支持)
PCIe ASPM (Active State Power Management): Disabled
PCIe ACS (Access Control Services): Enabled (透传需要)
Above 4G Decoding: Enabled
Resizable BAR: Enabled (支持的话)
```

**2. 验证 PCIe 配置**:
```bash
# 检查 PCIe 链路速度和宽度
lspci -vv | grep -A 10 "NVIDIA"

# 查看当前 PCIe 带宽
nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv

# 查看最大 PCIe 带宽
nvidia-smi --query-gpu=pcie.link.gen.max,pcie.link.width.max --format=csv
```

**3. Linux 内核参数**:
```bash
# 禁用 PCIe ASPM
GRUB_CMDLINE_LINUX_DEFAULT="pcie_aspm=off"
```

### 2.7 内存分配器优化

#### 使用高性能内存分配器

**1. Jemalloc**:
```bash
# 安装
apt-get install libjemalloc-dev

# 使用
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so python train.py

# 永久设置
echo 'export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so' >> ~/.bashrc
```

**2. TCMalloc**:
```bash
# 安装
apt-get install libgoogle-perftools-dev

# 使用
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so python train.py
```

**性能提升**: Jemalloc 和 TCMalloc 对深度学习工作负载可提升 10-30% 的性能。

### 2.8 CPU 功耗和性能平衡

#### NVIDIA Dynamic Boost (笔记本)

**nvidia-powerd daemon**:
- 动态在 GPU 和 CPU 间重新分配功率
- 基于工作负载需求自动调整
- 适用于支持的笔记本平台

#### 服务器平台

**1. Power Profile**:
```
BIOS Power Profile: Maximum Performance
Operating System Power Mode: High Performance
```

**2. 禁用节能功能**:
```
Intel SpeedStep: Disabled
AMD Cool'n'Quiet: Disabled
```

---

## 三、系统验证检查项

### 3.1 CPU 配置验证清单

#### 必检项目

| 检查项 | 命令 | 推荐值 |
|--------|------|--------|
| CPU Governor | `cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor` | performance |
| Turbo Boost (Intel) | `cat /sys/devices/system/cpu/intel_pstate/no_turbo` | 0 (enabled) |
| Turbo Core (AMD) | `cat /sys/devices/system/cpu/cpufreq/boost` | 1 (enabled) |
| NUMA 模式 | `numactl --hardware` | 多节点可见 |
| C-States | `cpupower idle-info` | C1 only 或 disabled |
| IOMMU | `dmesg \| grep -i iommu` | Enabled |
| PCIe ACS | `dmesg \| grep ACS` | Enabled |
| 当前频率 | `cpupower frequency-info` | 接近最大频率 |

### 3.2 NUMA 验证

```bash
#!/bin/bash
# NUMA 配置验证脚本

echo "=== NUMA Configuration Check ==="

# 1. 检查 NUMA 节点数
numa_nodes=$(numactl --hardware | grep "available:" | awk '{print $2}')
echo "NUMA Nodes: $numa_nodes"

# 2. 显示 NUMA 拓扑
echo -e "\nNUMA Topology:"
numactl --hardware

# 3. 检查 GPU 的 NUMA 亲和性
echo -e "\nGPU NUMA Affinity:"
nvidia-smi topo -m

# 4. 验证 GPU 到 NUMA 节点的映射
for gpu in $(nvidia-smi -L | awk '{print $2}' | tr -d ':'); do
    pci_id=$(nvidia-smi --id=$gpu --query-gpu=pci.bus_id --format=csv,noheader)
    # 转换为 sysfs 格式
    pci_path=$(echo $pci_id | tr '[:upper:]' '[:lower:]' | sed 's/://g' | sed 's/\./:/1')
    numa_node=$(cat /sys/bus/pci/devices/0000:$pci_path/numa_node 2>/dev/null || echo "N/A")
    echo "GPU $gpu (PCI: $pci_id): NUMA Node $numa_node"
done
```

### 3.3 CPU 频率和性能验证

```bash
#!/bin/bash
# CPU 性能配置验证

echo "=== CPU Performance Configuration Check ==="

# 1. CPU Governor
echo -e "\nCPU Governors:"
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | sort | uniq -c

# 2. CPU 频率
echo -e "\nCPU Frequencies:"
cpupower frequency-info | grep -E "current CPU frequency|boost state support"

# 3. Turbo Boost Status
echo -e "\nTurbo Boost Status:"
if [ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
    turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)
    [ $turbo -eq 0 ] && echo "Intel Turbo Boost: ENABLED" || echo "Intel Turbo Boost: DISABLED"
elif [ -f /sys/devices/system/cpu/cpufreq/boost ]; then
    boost=$(cat /sys/devices/system/cpu/cpufreq/boost)
    [ $boost -eq 1 ] && echo "AMD Turbo Core: ENABLED" || echo "AMD Turbo Core: DISABLED"
fi

# 4. C-States
echo -e "\nC-States:"
cpupower idle-info | grep -E "Number of idle states|Available idle states"
```

### 3.4 IOMMU 和 PCIe 验证

```bash
#!/bin/bash
# IOMMU 和 PCIe 配置验证

echo "=== IOMMU and PCIe Configuration Check ==="

# 1. IOMMU 状态
echo -e "\nIOMMU Status:"
if dmesg | grep -q "IOMMU enabled"; then
    echo "IOMMU: ENABLED"
    dmesg | grep -i "IOMMU" | tail -5
else
    echo "IOMMU: DISABLED or NOT FOUND"
fi

# 2. IOMMU 组
echo -e "\nIOMMU Groups:"
for d in /sys/kernel/iommu_groups/*/devices/*; do
    if [[ -e $d ]]; then
        group=$(echo $d | awk -F'/' '{print $(NF-2)}')
        device=$(basename $d)
        lspci -nns $device | grep -q NVIDIA && echo "Group $group: $(lspci -nns $device)"
    fi
done

# 3. PCIe 链路状态
echo -e "\nGPU PCIe Link Status:"
nvidia-smi --query-gpu=name,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max --format=csv

# 4. PCIe 错误检查
echo -e "\nPCIe Errors (if any):"
nvidia-smi --query-gpu=pci.replay_counter,pci.tx_util,pci.rx_util --format=csv
```

### 3.5 内存配置验证

```bash
#!/bin/bash
# 内存配置验证

echo "=== Memory Configuration Check ==="

# 1. Transparent Huge Pages
echo -e "\nTransparent Huge Pages:"
cat /sys/kernel/mm/transparent_hugepage/enabled

# 2. Swappiness
echo -e "\nSwappiness:"
cat /proc/sys/vm/swappiness

# 3. Dirty Ratio
echo -e "\nDirty Ratios:"
echo "dirty_ratio: $(cat /proc/sys/vm/dirty_ratio)"
echo "dirty_background_ratio: $(cat /proc/sys/vm/dirty_background_ratio)"
```

---

## 四、完整的 BIOS 配置推荐

### 4.1 CPU 配置

```
[CPU Configuration]
Intel Hyper-Threading: Enabled
AMD SMT (Simultaneous Multi-Threading): Enabled
Intel Turbo Boost Technology: Enabled
AMD Core Performance Boost: Enabled
Intel SpeedStep: Disabled
AMD Cool'n'Quiet: Disabled
C-States: Disabled (或 C1E Only)
Package C State: C0/C1 state
CPU Power and Performance Policy: Maximum Performance

[Advanced Power Management]
Power Technology: Custom
Energy/Performance Bias: Performance
CPU P State Control: Disabled (锁定在最高频率)
Hardware P-States: Disabled
```

### 4.2 内存配置

```
[Memory Configuration]
NUMA: Enabled
NUMA Nodes per Socket: 4 (AMD EPYC, HPC 工作负载)
Memory Interleaving: Disabled
Memory Operating Mode: Performance Mode
Patrol Scrub: Disabled (性能优先)
```

### 4.3 PCIe 和 I/O 配置

```
[PCIe Configuration]
PCIe Link Speed: Gen 4 / Gen 5 (Max)
PCIe ASPM Support: Disabled
PCIe ACS (Access Control Services): Enabled
Above 4G Decoding: Enabled
Resizable BAR Support: Enabled
MMIO High Base: 56T (或更高)
MMIO High Granularity Size: 256G (或更大)

[I/O Configuration]
VT-d (Intel) / IOMMU (AMD): Enabled
SR-IOV Support: Enabled
Interrupt Remapping: Enabled
```

### 4.4 其他优化

```
[Boot Options]
Quiet Boot: Disabled (查看启动信息)

[Miscellaneous]
Secure Boot: Disabled (可能与某些驱动冲突)
```

---

## 五、完整配置示例脚本

### 5.1 系统优化脚本

```bash
#!/bin/bash
# GPU 服务器性能优化脚本
# 适用于 Ubuntu 22.04 / RHEL 8+

set -e

echo "Starting GPU Server Performance Optimization..."

# 1. 设置 CPU Governor 为 performance
echo "Setting CPU governor to performance..."
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance > $cpu 2>/dev/null || true
done

# 2. 启用 Turbo Boost
echo "Enabling Turbo Boost..."
if [ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
    echo 0 > /sys/devices/system/cpu/intel_pstate/no_turbo
elif [ -f /sys/devices/system/cpu/cpufreq/boost ]; then
    echo 1 > /sys/devices/system/cpu/cpufreq/boost
fi

# 3. 禁用 CPU 空闲状态 (可选，降低延迟)
# echo 1 > /sys/devices/system/cpu/cpu*/cpuidle/state*/disable

# 4. 设置 Transparent Huge Pages 为 always
echo always > /sys/kernel/mm/transparent_hugepage/enabled
echo always > /sys/kernel/mm/transparent_hugepage/defrag

# 5. 调整 swappiness
sysctl -w vm.swappiness=10

# 6. 设置网络优化（如果需要）
sysctl -w net.core.rmem_max=268435456
sysctl -w net.core.wmem_max=268435456

# 7. 创建 systemd 服务使配置持久化
cat > /etc/systemd/system/gpu-performance-tuning.service <<EOF
[Unit]
Description=GPU Server Performance Tuning
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/gpu-perf-tune.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

systemctl enable gpu-performance-tuning.service

echo "GPU Server Performance Optimization Complete!"
```

---

## 六、总结和建议

### 6.1 关键要点

1. **使用最新工具**:
   - NVIDIA DeepOps (裸金属集群)
   - NVIDIA GPU Operator (Kubernetes)
   - NVIDIA Dynamo (2025 年新工具，AI 推理)

2. **CPU 优化是关键**:
   - NUMA 配置直接影响 GPU 性能
   - CPU Governor 必须设置为 performance
   - Turbo Boost 必须启用
   - C-States 应禁用或仅启用 C1

3. **系统验证必不可少**:
   - 检查 NUMA 配置和 GPU 亲和性
   - 验证 PCIe 链路状态
   - 确认 IOMMU 正确启用
   - 监控 CPU 频率和性能状态

### 6.2 性能提升预期

正确配置后的性能提升：
- NUMA 优化: 20-40%
- CPU Governor + Turbo: 10-20%
- 内存分配器优化: 10-30%
- PCIe 优化: 5-15%

### 6.3 推荐工具栈 (2025)

**裸金属部署**:
```
NVIDIA DeepOps
├── Ansible (配置管理)
├── Slurm (HPC 工作负载)
└── GPU 优化配置
```

**Kubernetes 部署**:
```
NVIDIA GPU Operator
├── NVIDIA Device Plugin
├── NVIDIA Container Toolkit
├── DCGM Exporter
└── Node Feature Discovery
```

**AI 推理平台**:
```
NVIDIA Dynamo
├── Kubernetes Operator
├── Auto-scaler
└── 动态资源管理
```

---

## 参考资料

1. NVIDIA DeepOps: https://github.com/NVIDIA/deepops
2. NVIDIA GPU Operator: https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/
3. PyTorch Performance Tuning: https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html
4. NVIDIA Triton Optimization: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/optimization.html
5. AMD ROCm System Optimization: https://rocm.docs.amd.com/en/latest/how-to/system-optimization/

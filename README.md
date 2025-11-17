# GPU Passthrough - 基线安装与验证自动化 (2025 版本)

这个项目提供基于 Ansible 的 GPU 机器基线安装和验证自动化解决方案，基于 NVIDIA 官方工具和 2024-2025 年最新的开源社区最佳实践。

> **🆕 2025 年更新**: 新增 CPU 性能优化、NUMA 配置、完整系统验证、通讯带宽测试和模型训练基准测试

## 项目目标

1. **自动化安装**: 通过 Ansible 自动化安装 GPU 机器的基线环境（驱动、CUDA、容器运行时）
2. **CPU 性能优化**: 优化 CPU 配置以最大化 GPU 工作负载性能（NUMA、频率调节、Turbo Boost 等）
3. **全面验证**: 提供多级别的验证脚本，检查 CPU、GPU、NUMA、IOMMU、PCIe 等所有配置
4. **🆕 通讯带宽测试**: PCIe、NVLink、RDMA 带宽测试，与性能基线对比
5. **🆕 模型训练基准**: NCCL 集合通信测试、Megatron-LM 训练吞吐量测试
6. **开源整合**: 基于 NVIDIA DeepOps、GPU Operator 等 2024-2025 年最新工具和最佳实践

## 项目结构

```
gpu_passthrough/
├── ansible/                    # Ansible 自动化配置
│   ├── roles/
│   │   ├── gpu_baseline/      # GPU 基线安装 role
│   │   ├── cpu_optimization/  # 🆕 CPU 性能优化 role
│   │   ├── benchmark_tools/   # 🆕 基准测试工具 role
│   │   └── gpu_validation/    # GPU 验证 role
│   ├── playbooks/
│   │   ├── setup_gpu_baseline.yml           # GPU 基线安装
│   │   ├── full_deployment_optimized.yml    # 🆕 完整优化部署
│   │   └── validate_gpu.yml                 # GPU 验证
│   ├── inventory/             # 主机清单
│   └── ansible.cfg
├── scripts/                    # 验证和监控脚本
│   ├── validation/
│   │   ├── quick_check.sh     # 快速验证
│   │   ├── system_check.sh    # 🆕 全面系统验证
│   │   ├── bandwidth_test.sh  # 🆕 带宽测试
│   │   └── gpu_health.py      # GPU 健康检查
│   ├── benchmarks/            # 🆕 基准测试
│   │   ├── nccl_benchmark.sh  # NCCL 测试
│   │   └── megatron_benchmark.sh # Megatron 训练基准
│   ├── utils/                 # 工具脚本
│   │   └── performance_baselines.py # 性能基线数据库
│   └── monitoring/            # 监控脚本
├── docs/                       # 文档
│   ├── research.md            # 开源项目调研报告
│   ├── latest_research_2025.md # 🆕 2024-2025 最新调研
│   ├── bandwidth_and_benchmarks.md # 🆕 带宽测试和基准测试指南
│   └── implementation_plan.md # 实施方案
└── README.md
```

## 核心功能

### 1. GPU 基线安装 (gpu_baseline role)

自动化安装以下组件：

- ✅ NVIDIA GPU 驱动
- ✅ CUDA Toolkit
- ✅ NVIDIA Container Toolkit (Docker/containerd)
- ✅ GPU 配置优化（持久化模式、功率限制等）

**基于的开源项目**:
- [NVIDIA/ansible-role-nvidia-driver](https://github.com/NVIDIA/ansible-role-nvidia-driver)
- [NVIDIA/ansible-role-nvidia-docker](https://github.com/NVIDIA/ansible-role-nvidia-docker)
- [CSCfi/ansible-role-cuda](https://github.com/fgci-org/ansible-role-cuda)
- [datadrivers/ansible-role-docker](https://github.com/datadrivers/ansible-role-docker)

### 2. 🆕 CPU 性能优化 (cpu_optimization role)

**针对 GPU 工作负载优化 CPU 配置**，性能提升 20-40%：

- ✅ CPU Governor 设置（Performance 模式）
- ✅ Turbo Boost / Turbo Core 启用
- ✅ NUMA 优化和亲和性配置
- ✅ C-States 优化（降低延迟）
- ✅ IOMMU 配置（Intel VT-d / AMD-Vi）
- ✅ PCIe 性能优化
- ✅ Transparent Huge Pages 配置
- ✅ 内存参数调优（swappiness、dirty ratio）

**关键优化项**:
| 优化项 | 默认值 | 优化值 | 性能提升 |
|--------|--------|--------|----------|
| CPU Governor | ondemand/powersave | performance | 10-20% |
| NUMA 配置 | 自动 | 绑定到对应节点 | 20-40% |
| Turbo Boost | 未知 | 强制启用 | 10-15% |
| C-States | 深度睡眠 | C1 only | 5-10% (降低延迟) |

**基于最佳实践**:
- NVIDIA DeepOps 配置
- PyTorch/TensorFlow 性能调优指南
- AMD ROCm 系统优化文档

### 3. 🆕 全面系统验证 (system_check.sh)

**完整的系统配置验证**，涵盖 8 大类检查项：

1. **CPU 配置**: Governor、Turbo Boost、频率、C-States
2. **NUMA 配置**: 节点数、GPU 亲和性、拓扑结构
3. **IOMMU 配置**: VT-d/AMD-Vi 启用、IOMMU 组、内核参数
4. **PCIe 配置**: 链路速度、宽度、错误计数、ACS
5. **GPU 配置**: 驱动版本、持久化模式、温度、ECC 错误
6. **内存配置**: THP、swappiness、dirty ratio
7. **内核参数**: GRUB 配置检查
8. **容器运行时**: Docker/containerd GPU 访问测试

**验证输出**:
- JSON 格式详细报告
- 彩色终端输出（✓ 通过 / ⚠ 警告 / ✗ 失败）
- 自动化可集成到 CI/CD

### 4. 🆕 通讯带宽测试 (bandwidth_test.sh)

**完整的通讯带宽测试和性能基线对比**：

#### 机内通讯测试
- ✅ **PCIe 带宽**: Host-Device 和 Device-Host 传输（使用 nvbandwidth, bandwidthTest）
- ✅ **NVLink 带宽**: GPU-GPU P2P 传输（使用 p2pBandwidthLatencyTest）
- ✅ **GPU 拓扑**: 自动检测 NVLink 连接和 PCIe 配置

#### 机间通讯测试
- ✅ **RDMA 带宽**: InfiniBand/RoCE 网络性能（使用 ib_write_bw）
- ✅ **GPUDirect RDMA**: GPU 直接访问远程 GPU 内存
- ✅ **网络拓扑**: 自动检测 IB 设备和链路速度

#### 性能基线数据库

| GPU 型号 | 内存带宽 | NVLink BW | PCIe BW | AllReduce (8GPU) |
|---------|---------|-----------|---------|------------------|
| A100-SXM4-80GB | 2039 GB/s | 600 GB/s | 64 GB/s | 250 GB/s |
| H100-SXM5-80GB | 3350 GB/s | 900 GB/s | 128 GB/s | 450 GB/s |
| V100-SXM2-32GB | 900 GB/s | 300 GB/s | 32 GB/s | 180 GB/s |

**使用方式**:
```bash
# 运行完整带宽测试
./scripts/validation/bandwidth_test.sh

# 或使用快捷命令（安装后）
gpu-benchmark bandwidth

# 查看性能基线
python3 scripts/utils/performance_baselines.py list
python3 scripts/utils/performance_baselines.py info A100-SXM4-80GB
```

### 5. 🆕 NCCL 集合通信测试 (nccl_benchmark.sh)

**测试分布式训练的集合通信性能**：

- ✅ **AllReduce**: 最常用的梯度同步操作
- ✅ **Broadcast**: 参数广播性能
- ✅ **Reduce-Scatter**: 分布式 reduce 操作
- ✅ **All-Gather**: 收集操作性能
- ✅ **多节点支持**: MPI 集成，支持跨节点测试
- ✅ **性能基线对比**: 自动对比预期性能

**预期性能（Bus Bandwidth）**:
- **A100 8-GPU 节点内**: ~250 GB/s
- **H100 8-GPU 节点内**: ~450 GB/s
- **A100 跨节点 (IB HDR)**: ~180 GB/s

**使用方式**:
```bash
# 单节点 NCCL 测试
./scripts/benchmarks/nccl_benchmark.sh

# 或使用快捷命令
gpu-benchmark nccl

# 多节点测试（需要 MPI）
mpirun -np 64 -N 8 --hostfile hosts \
    /opt/nccl-tests/build/all_reduce_perf -b 8 -e 8G -f 2 -g 1
```

### 6. 🆕 Megatron-LM 训练基准 (megatron_benchmark.sh)

**实际模型训练性能测试**：

- ✅ **GPT 模型训练**: 支持 GPT-1.2B, GPT-8.3B, GPT-175B
- ✅ **TFLOPS 测量**: 实际计算吞吐量
- ✅ **MFU 计算**: Model FLOP Utilization（模型利用率）
- ✅ **扩展性测试**: 多 GPU/多节点性能
- ✅ **性能基线对比**: 与已知基准对比

**性能基线 (GPT-1.2B 单 GPU)**:

| GPU 型号 | TFLOPS | MFU | Samples/sec |
|---------|--------|-----|-------------|
| V100 | 39 | 30% | 12 |
| A100 | 93.6 | 60% | 28 |
| H100 | 178 | 47% | 45 |

**使用方式**:
```bash
# 运行 GPT-1.2B 基准测试
MODEL_SIZE=GPT-1.2B ./scripts/benchmarks/megatron_benchmark.sh

# 或使用快捷命令
MODEL_SIZE=GPT-1.2B gpu-benchmark megatron

# 自定义参数
MODEL_SIZE=GPT-8.3B BATCH_SIZE=16 NUM_STEPS=200 \
    gpu-benchmark megatron
```

### 7. GPU 验证测试 (多级别)

#### Level 1: 快速验证 (1-5 分钟)
- nvidia-smi 可用性检查
- GPU 设备检测
- 驱动和 CUDA 版本确认
- 基础健康检查（温度、内存、PCIe）

#### Level 2: 标准验证 (10-15 分钟)
- Level 1 所有检查
- DCGM 快速诊断
- CUDA 功能测试
- 容器 GPU 访问测试
- 内存带宽测试

#### Level 3: 完整验证 (30-60 分钟)
- Level 2 所有检查
- DCGM 完整诊断套件
- GPU-Burn 压力测试
- 长时间稳定性测试
- 性能基准测试

**基于的工具**:
- [NVIDIA DCGM](https://github.com/NVIDIA/DCGM) - 数据中心 GPU 管理器
- [NVIDIA Validation Suite (NVVS)](https://docs.nvidia.com/deploy/nvvs-user-guide/)
- [GPU-Burn](https://github.com/wilicc/gpu-burn) - GPU 压力测试
- [gpustat](https://github.com/wookayin/gpustat) - GPU 状态监控
- [nvitop](https://github.com/XuehaiPan/nvitop) - GPU 进程管理

## 快速开始

### 前置要求

**控制节点**:
- Ansible >= 2.10
- Python >= 3.8
- SSH 访问目标主机

**目标主机**:
- Ubuntu 20.04/22.04 或 RHEL/CentOS 8+
- 至少 10GB 可用磁盘空间
- NVIDIA GPU 硬件
- 管理员权限

### 安装步骤

#### 1. 克隆项目

```bash
git clone <repository-url>
cd gpu_passthrough
```

#### 2. 配置主机清单

编辑 `ansible/inventory/hosts.yml`:

```yaml
all:
  children:
    gpu_nodes:
      hosts:
        gpu-server-01:
          ansible_host: 192.168.1.101
          ansible_user: ubuntu
```

#### 3. 配置变量

编辑 `ansible/inventory/group_vars/gpu_nodes.yml` 根据需求调整配置：

```yaml
# GPU 配置
nvidia_driver_version: "535"
cuda_version: "12-2"
install_cuda: true
install_container_runtime: true
container_runtime: "docker"

# 🆕 CPU 优化配置
cpu_governor: "performance"
enable_turbo_boost: true
optimize_numa: true
vm_swappiness: 10
```

#### 4. 🆕 完整优化部署（推荐）

```bash
cd ansible

# 完整部署：GPU 基线 + CPU 优化
ansible-playbook playbooks/full_deployment_optimized.yml

# 部署后需要重启系统
ssh gpu-server-01 sudo reboot
```

#### 5. 🆕 运行全面系统验证

```bash
# 方法 1: 通过 Ansible playbook
ansible-playbook playbooks/validate_gpu.yml -e "level=standard"

# 方法 2: 直接运行验证脚本（在目标主机上）
ssh gpu-server-01
sudo /path/to/scripts/validation/system_check.sh

# 方法 3: 快速 GPU 检查
./scripts/validation/quick_check.sh /tmp/gpu_check.json
```

#### 旧方式：仅 GPU 基线安装（不含 CPU 优化）

```bash
cd ansible
ansible-playbook playbooks/setup_gpu_baseline.yml
```

### 单独使用验证脚本

```bash
# 🆕 全面系统检查（推荐）
sudo ./scripts/validation/system_check.sh /tmp/system_check.json

# GPU 快速检查
./scripts/validation/quick_check.sh /tmp/gpu_check.json

# Python GPU 健康检查
python3 scripts/validation/gpu_health.py -o /tmp/health_report.json -v

# 🆕 查看 NUMA 和 GPU 亲和性
sudo numa-gpu-info  # 部署后自动安装
```

## 配置说明

### 关键变量

#### GPU 配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `nvidia_driver_version` | "535" | NVIDIA 驱动版本 |
| `cuda_version` | "12-2" | CUDA Toolkit 版本 |
| `install_cuda` | true | 是否安装 CUDA |
| `install_container_runtime` | true | 是否安装容器运行时 |
| `container_runtime` | "docker" | 容器运行时类型 (docker/containerd) |
| `gpu_persistence_mode` | true | GPU 持久化模式 |
| `validation_level` | "quick" | 验证级别 (quick/standard/full) |

#### 🆕 CPU 优化配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `cpu_governor` | "performance" | CPU 频率调节器 |
| `enable_turbo_boost` | true | 启用 Turbo Boost/Turbo Core |
| `optimize_numa` | true | 启用 NUMA 优化 |
| `disable_deep_cstates` | true | 禁用深度 C-States |
| `max_cstate` | 1 | 最大 C-State (C0/C1 only) |
| `thp_enabled` | "always" | Transparent Huge Pages |
| `vm_swappiness` | 10 | 内存交换倾向值 |
| `install_perf_service` | true | 安装性能调优服务 |

### 自定义配置

可以在以下位置覆盖默认配置：

1. `ansible/inventory/group_vars/gpu_nodes.yml` - 组级别变量
2. `ansible/inventory/hosts.yml` - 主机级别变量
3. 命令行参数: `-e "variable=value"`

## 验证报告

验证完成后，报告会保存在：

- **目标主机**: `/tmp/gpu_validation/`
- **控制节点**: `./validation_results/<hostname>/`

报告格式：
- JSON 格式：详细的结构化数据
- 文本格式：可读的验证摘要
- HTML 格式：可视化报告（完整验证）

## 故障排除

### 常见问题

**1. nvidia-smi 不可用**
```bash
# 检查驱动是否加载
lsmod | grep nvidia

# 检查 nouveau 是否被禁用
lsmod | grep nouveau

# 重新运行基线安装
ansible-playbook playbooks/setup_gpu_baseline.yml
```

**2. 容器无法访问 GPU**
```bash
# 检查 NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi

# 检查 Docker daemon 配置
cat /etc/docker/daemon.json
```

**3. GPU 温度过高**
```bash
# 检查散热
nvidia-smi -q -d TEMPERATURE

# 调整功率限制
nvidia-smi -pl 250  # 设置为 250W
```

## 🆕 2024-2025 最新工具和技术

### NVIDIA 官方最新工具

#### NVIDIA DeepOps (裸金属集群部署)
- **项目**: https://github.com/NVIDIA/deepops
- **版本**: 22.04.1 (持续维护)
- **用途**: GPU 集群部署最佳实践，支持 Kubernetes 和 Slurm
- **特点**: 裸金属优化、DGX 系统支持、完整监控栈

#### NVIDIA GPU Operator (Kubernetes)
- **2024-2025 最活跃项目**
- **用途**: Kubernetes GPU 管理标准化
- **特性**: vGPU、MIG、Time Slicing、GPUDirect RDMA/Storage

#### NVIDIA Dynamo (2025 年新工具)
- **发布**: 2025 年初
- **创新**: AI 推理感知的自动扩展器
- **特性**: 单命令部署到数千 GPU、动态资源管理

#### NVIDIA KAI Scheduler (2025 年开源)
- **发布**: 2025 年 1 月
- **用途**: 企业级 GPU 调度器
- **特点**: Kubernetes AI 工作负载优化

### CPU 优化参考资源

- **PyTorch Performance Tuning**: https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html
- **NVIDIA Triton Optimization**: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/optimization.html
- **AMD ROCm System Optimization**: https://rocm.docs.amd.com/en/latest/how-to/system-optimization/
- **Intel VTune NUMA Analysis**: https://www.intel.com/content/www/us/en/docs/vtune-profiler/cookbook/

## 开源项目参考

本项目整合了以下优秀的开源项目和工具：

### Ansible Roles
- [NVIDIA/ansible-role-nvidia-driver](https://github.com/NVIDIA/ansible-role-nvidia-driver)
- [NVIDIA/ansible-role-nvidia-docker](https://github.com/NVIDIA/ansible-role-nvidia-docker)
- [NVIDIA/deepops](https://github.com/NVIDIA/deepops) - 🆕 2024-2025 推荐
- [CSCfi/ansible-role-cuda](https://github.com/fgci-org/ansible-role-cuda)
- [Provizanta/ansible-role-nvidia-cuda](https://github.com/Provizanta/ansible-role-nvidia-cuda)
- [datadrivers/ansible-role-docker](https://github.com/datadrivers/ansible-role-docker)

### 验证和监控工具
- [NVIDIA DCGM](https://github.com/NVIDIA/DCGM)
- [NVIDIA GPU Stress Test](https://github.com/NVIDIA/GPUStressTest)
- [GPU-Burn](https://github.com/wilicc/gpu-burn)
- [gpustat](https://github.com/wookayin/gpustat)
- [nvitop](https://github.com/XuehaiPan/nvitop)
- [GPUtil](https://github.com/anderskm/gputil)

### 文档和指南
- [NVIDIA Validation Suite User Guide](https://docs.nvidia.com/deploy/nvvs-user-guide/)
- [DCGM User Guide](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/)
- [NVIDIA GPU Operator Docs](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/)

## 文档

- [开源项目调研报告](docs/research.md) - 详细的开源项目调研和分析
- [🆕 2024-2025 最新调研](docs/latest_research_2025.md) - 最新工具、CPU 优化、BIOS 配置完整指南
- [实施方案](docs/implementation_plan.md) - 完整的技术实施方案

## 🆕 关键 BIOS 配置建议

基于 2024-2025 最佳实践，以下 BIOS 设置可显著提升 GPU 性能：

### CPU 配置
```
Intel Hyper-Threading / AMD SMT: Enabled
Intel Turbo Boost / AMD Core Performance Boost: Enabled
Intel SpeedStep / AMD Cool'n'Quiet: Disabled
C-States: Disabled (或 C1E Only)
CPU Power Policy: Maximum Performance
```

### 内存配置
```
NUMA: Enabled
NUMA Nodes per Socket (NPS): 4 (AMD EPYC, HPC workloads)
Memory Interleaving: Disabled
```

### PCIe 和 I/O
```
PCIe Link Speed: Gen 4 / Gen 5 (Max)
PCIe ASPM: Disabled
PCIe ACS: Enabled
VT-d (Intel) / IOMMU (AMD): Enabled
Above 4G Decoding: Enabled
Resizable BAR: Enabled
SR-IOV Support: Enabled
```

详细 BIOS 配置指南请参考: [docs/latest_research_2025.md](docs/latest_research_2025.md#四完整的-bios-配置推荐)

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

本项目基于 MIT 许可证开源。

## 致谢

感谢 NVIDIA 和开源社区提供的优秀工具和最佳实践。

---

**项目维护者**: 请根据实际情况更新

**最后更新**: 2025-01-15

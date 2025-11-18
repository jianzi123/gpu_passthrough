# GPU Passthrough - 基线安装与验证自动化 (2025 版本)

这个项目提供基于 Ansible 的 GPU 机器基线安装和验证自动化解决方案，基于 NVIDIA 官方工具和 2024-2025 年最新的开源社区最佳实践。

> **🆕 2025 年更新**: 新增 CPU 性能优化、NUMA 配置、完整系统验证、通讯带宽测试、模型训练基准测试、GPU-CUDA 兼容性自动匹配、NGC 容器镜像管理、**慢节点检测**

## 项目目标

1. **自动化安装**: 通过 Ansible 自动化安装 GPU 机器的基线环境（驱动、CUDA、容器运行时）
2. **🆕 智能 CUDA 选择**: 自动检测 GPU 型号并选择对应的 CUDA 和驱动版本
3. **CPU 性能优化**: 优化 CPU 配置以最大化 GPU 工作负载性能（NUMA、频率调节、Turbo Boost 等）
4. **全面验证**: 提供多级别的验证脚本，检查 CPU、GPU、NUMA、IOMMU、PCIe 等所有配置
5. **🆕 通讯带宽测试**: PCIe、NVLink、RDMA 带宽测试，与性能基线对比
6. **🆕 模型训练基准**: NCCL 集合通信测试、Megatron-LM 训练吞吐量测试
7. **🆕 NGC 容器管理**: 自动化拉取和管理 NVIDIA NGC 镜像（PyTorch、NeMo、Triton 等）
8. **🆕 慢节点检测**: 自动检测集群中性能异常的节点和 GPU（NVLink、PCIe、NCCL 通讯）
9. **开源整合**: 基于 NVIDIA DeepOps、GPU Operator 等 2024-2025 年最新工具和最佳实践

## 项目结构

```
gpu_passthrough/
├── ansible/                    # Ansible 自动化配置
│   ├── roles/
│   │   ├── gpu_baseline/      # GPU 基线安装 role (含 GPU 自动检测)
│   │   ├── cpu_optimization/  # 🆕 CPU 性能优化 role
│   │   ├── benchmark_tools/   # 🆕 基准测试工具 role
│   │   ├── ngc_images/        # 🆕 NGC 容器镜像管理 role
│   │   ├── slow_node_detection/ # 🆕 慢节点检测 role
│   │   └── gpu_validation/    # GPU 验证 role
│   ├── playbooks/
│   │   ├── setup_gpu_baseline.yml           # GPU 基线安装
│   │   ├── full_deployment_optimized.yml    # 🆕 完整优化部署
│   │   ├── detect_slow_nodes.yml            # 🆕 慢节点检测
│   │   └── validate_gpu.yml                 # GPU 验证
│   ├── inventory/             # 主机清单
│   └── ansible.cfg
├── scripts/                    # 验证和监控脚本
│   ├── install/               # 🆕 安装脚本
│   │   ├── install_gpu_driver.sh  # 🆕 GPU 驱动安装脚本（多方法支持）
│   │   └── build_precompiled_driver.sh # 🆕 预编译驱动构建脚本
│   ├── validation/
│   │   ├── quick_check.sh     # 快速验证
│   │   ├── system_check.sh    # 🆕 全面系统验证
│   │   ├── rdma_check.sh      # 🆕 RDMA 环境验证
│   │   ├── bandwidth_test.sh  # 🆕 带宽测试
│   │   ├── intra_node_bandwidth_check.sh   # 🆕 节点内部带宽检查
│   │   ├── inter_node_nccl_check.sh        # 🆕 跨节点 NCCL 通讯检查
│   │   ├── detect_slow_nodes.sh            # 🆕 综合慢节点检测工具
│   │   └── gpu_health.py      # GPU 健康检查
│   ├── benchmarks/            # 🆕 基准测试
│   │   ├── nccl_benchmark.sh  # NCCL 测试
│   │   └── megatron_benchmark.sh # Megatron 训练基准
│   ├── utils/                 # 工具脚本
│   │   ├── performance_baselines.py # 性能基线数据库
│   │   ├── cuda_compatibility.py    # 🆕 GPU-CUDA 兼容性数据库
│   │   ├── ngc_images.py            # 🆕 NGC 镜像注册表
│   │   ├── ngc_manager.sh           # 🆕 NGC 镜像管理工具
│   │   └── manage_precompiled_drivers.sh # 🆕 预编译驱动管理工具
│   └── monitoring/            # 监控脚本
├── docs/                       # 文档
│   ├── research.md            # 开源项目调研报告
│   ├── latest_research_2025.md # 🆕 2024-2025 最新调研
│   ├── bandwidth_and_benchmarks.md # 🆕 带宽测试和基准测试指南
│   ├── cuda_compatibility_and_ngc.md # 🆕 CUDA 兼容性和 NGC 镜像指南
│   ├── gpu_driver_installation_methods.md # 🆕 GPU 驱动安装方法指南
│   ├── precompiled_driver_guide.md  # 🆕 预编译驱动完整指南
│   ├── slow_node_detection.md       # 🆕 慢节点检测完整指南
│   ├── best_practices.md      # 🆕 最佳实践指南
│   ├── QUICKSTART.md          # 🆕 快速开始指南
│   └── implementation_plan.md # 实施方案
└── README.md
```

## 核心功能

### 1. GPU 基线安装 (gpu_baseline role)

自动化安装以下组件：

- ✅ NVIDIA GPU 驱动（支持多种安装方法）
- ✅ CUDA Toolkit
- ✅ NVIDIA Container Toolkit (Docker/containerd)
- ✅ GPU 配置优化（持久化模式、功率限制等）
- ✅ 🆕 **GPU 自动检测**: 自动识别 GPU 型号并选择对应的 CUDA 和驱动版本
- ✅ 🆕 **多种安装方法**: Native、Driver Container、Precompiled

**🆕 GPU-CUDA 兼容性自动匹配**:

支持的 GPU 型号和自动选择的 CUDA 版本：

| GPU 架构 | GPU 型号 | 推荐 CUDA | 推荐驱动 |
|---------|---------|-----------|----------|
| Volta | V100 | 12.2 | 535.154.05 |
| Ampere | A100, A800, RTX 3090 | 12.2 | 535.154.05 |
| Hopper | H100, H800 | 12.3 | 545.23.08 |
| Ada Lovelace | RTX 4090 | 12.2 | 535.154.05 |

```bash
# 启用自动检测（默认启用）
auto_detect_cuda_version: true

# Ansible 会自动：
# 1. 检测 GPU 型号（lspci）
# 2. 查询兼容性数据库
# 3. 选择推荐的 CUDA 版本和驱动版本
# 4. 记录检测报告到 /var/log/gpu_baseline/gpu_detection.txt
```

**🆕 GPU 驱动安装方法**:

支持三种驱动安装方式，基于 NVIDIA GPU Operator 架构：

| 方法 | 特点 | 适用场景 |
|------|------|----------|
| **Native** | 传统安装，直接在系统上安装驱动 | 传统数据中心、物理服务器 |
| **Driver Container** | 容器化驱动，基于 GPU Operator | Kubernetes、云原生环境 |
| **Precompiled** | 预编译驱动，快速部署 | 大规模部署、内核统一环境 |

```bash
# 方法 1: Native 安装（默认）
ansible-playbook -i inventory/hosts playbooks/setup_gpu_baseline.yml \
  -e "driver_installation_method=native"

# 方法 2: Driver Container 安装
ansible-playbook -i inventory/hosts playbooks/setup_gpu_baseline.yml \
  -e "driver_installation_method=driver-container"

# 方法 3: 使用独立脚本安装
sudo ./scripts/install/install_gpu_driver.sh --method native --auto-detect
sudo ./scripts/install/install_gpu_driver.sh --method driver-container
sudo ./scripts/install/install_gpu_driver.sh --method precompiled
```

**详细文档**:
- [GPU 驱动安装方法指南](docs/gpu_driver_installation_methods.md) - 三种方法的详细对比和使用指南
- [预编译驱动完整指南](docs/precompiled_driver_guide.md) - 预编译驱动的构建、部署和管理

**🆕 预编译驱动特性**:

预编译驱动提供最快的部署速度和最低的资源消耗，特别适合大规模部署：

```bash
# 构建预编译驱动（一次性）
./scripts/install/build_precompiled_driver.sh \
    --driver-version 535.154.05 \
    --kernel-version 5.15.0-91-generic \
    --container-build

# 批量构建多个版本
./scripts/install/batch_build_drivers.sh

# 管理预编译驱动
./scripts/utils/manage_precompiled_drivers.sh list          # 列出所有可用驱动
./scripts/utils/manage_precompiled_drivers.sh install 535.154.05  # 安装指定版本
./scripts/utils/manage_precompiled_drivers.sh rollback      # 回滚到上一版本
```

**性能对比**（100 节点集群）:

| 指标 | 传统安装 | 预编译驱动 | 节省 |
|------|---------|-----------|------|
| 部署时间 | ~50 小时 | ~3.5 小时 | 93% |
| CPU 时间 | 100 分钟/节点 | 5 分钟/节点 | 95% |
| 内存使用 | 4 GB | 512 MB | 87% |
| 网络带宽 | 200 MB/节点 | 50 MB/节点 | 75% |

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
- ✅ **🆕 NGC 容器支持**: 支持使用 NGC NeMo 镜像运行

**性能基线 (GPT-1.2B 单 GPU)**:

| GPU 型号 | TFLOPS | MFU | Samples/sec |
|---------|--------|-----|-------------|
| V100 | 39 | 30% | 12 |
| A100 | 93.6 | 60% | 28 |
| H100 | 178 | 47% | 45 |

**使用方式**:
```bash
# 使用 NGC NeMo 容器运行（推荐）
export USE_NGC_CONTAINER=true
MODEL_SIZE=GPT-1.2B ./scripts/benchmarks/megatron_benchmark.sh

# 或使用本地 Megatron-LM
export USE_NGC_CONTAINER=false
MODEL_SIZE=GPT-1.2B ./scripts/benchmarks/megatron_benchmark.sh

# 自定义参数
MODEL_SIZE=GPT-8.3B BATCH_SIZE=16 NUM_STEPS=200 \
    gpu-benchmark megatron
```

### 7. 🆕 NGC 容器镜像管理 (ngc_images role)

**自动化管理 NVIDIA NGC (GPU Cloud) 容器镜像**：

NVIDIA NGC 提供预优化的深度学习和推理容器，包含 CUDA、cuDNN、NCCL 等完整工具链。

**支持的 NGC 镜像**:

| 镜像 | 版本 | 用途 | 主要组件 |
|------|------|------|----------|
| **pytorch** | 24.01 | 训练/推理 | PyTorch 2.3, CUDA 12.3, TensorRT 8.6 |
| **nemo** | 24.01 | LLM 训练 | Megatron-LM 0.5, NeMo 1.22, Transformer Engine |
| **triton** | 24.01 | 推理服务 | Triton Server 2.42, TensorRT, 多后端支持 |
| **tensorflow** | 24.01 | 训练/推理 | TensorFlow 2.15, CUDA 12.3 |
| **tensorrt** | 24.01 | 推理优化 | TensorRT 8.6, ONNX Parser |
| **cuda** | 12.3.2 | 开发 | CUDA Toolkit, NVCC, cuBLAS |
| **rapids** | 24.02 | 数据科学 | cuDF, cuML, cuGraph, Dask |
| **deepstream** | 6.4 | 视频分析 | DeepStream, Triton 集成 |

**功能特性**:

- ✅ **自动拉取**: 基于 CUDA 版本自动选择兼容镜像
- ✅ **GPU 测试**: 拉取后自动验证 GPU 功能
- ✅ **镜像管理**: 便捷的命令行工具管理镜像
- ✅ **清单报告**: 自动生成镜像清单和测试报告

**使用方式**:

```bash
# 查看可用 NGC 镜像
./scripts/utils/ngc_manager.sh list

# 拉取 PyTorch 镜像
./scripts/utils/ngc_manager.sh pull pytorch

# 拉取特定版本
./scripts/utils/ngc_manager.sh pull pytorch 24.01

# 运行镜像（交互式）
./scripts/utils/ngc_manager.sh run pytorch

# 测试镜像 GPU 功能
./scripts/utils/ngc_manager.sh test pytorch

# 查看 CUDA 12.3 兼容镜像
./scripts/utils/ngc_manager.sh cuda 12.3
```

**Ansible 自动化部署**:

```yaml
# ansible/roles/ngc_images/defaults/main.yml
ngc_images_to_pull:
  - name: pytorch
    version: "24.01"
  - name: nemo
    version: "24.01"
  - name: triton
    version: "24.01"

# 自动根据 CUDA 版本选择镜像
auto_select_images_by_cuda: true
```

运行 playbook:
```bash
ansible-playbook -i inventory/hosts playbooks/setup_ngc_images.yml
```

**NGC 镜像使用示例**:

```bash
# 使用 PyTorch 镜像训练
docker run --gpus all -it --rm \
  --ipc=host --network=host \
  -v $HOME/workspace:/workspace \
  nvcr.io/nvidia/pytorch:24.01-py3

# 使用 Triton 部署推理服务
docker run --gpus all -it --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v /path/to/models:/models \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models
```

### 8. 🆕 慢节点检测 (Slow Node Detection)

**自动检测 GPU 集群中性能异常的节点**，基于业界最佳实践（Microsoft Azure DGX Cloud、Together.AI）：

#### 检测方法

**1. 节点内部带宽检测** (`intra_node_bandwidth_check.sh`)
- ✅ **NVLink 拓扑和状态**: 检查 NVLink 连接是否 active，识别降速链路
- ✅ **GPU-GPU 带宽**: 使用 p2pBandwidthLatencyTest、nvbandwidth 测量 GPU 间带宽
- ✅ **PCIe 带宽**: Host-to-Device 和 Device-to-Host 传输性能
- ✅ **自动基线对比**: 与 A100/H100/V100 性能基线对比，识别慢 GPU

**2. 跨节点 NCCL 通讯检测** (`inter_node_nccl_check.sh`)
- ✅ **多次迭代统计**: 运行多次 NCCL all-reduce 测试，计算均值/标准差/最小值/最大值
- ✅ **成对测试 (Pairwise)**: 测试每对节点之间的通讯性能，识别问题节点对
- ✅ **二分搜索 (Binary Search)**: 快速定位慢节点（适用于 4+ 节点）
- ✅ **性能基线对比**: 与 NCCL 性能基线对比，检测低于阈值的节点

**3. 综合检测工具** (`detect_slow_nodes.sh`)
- ✅ **统一界面**: 整合节点内部和跨节点检测
- ✅ **并行执行**: 支持并行运行节点内部检查（更快）
- ✅ **自动化报告**: 生成综合报告，识别所有问题节点和 GPU
- ✅ **Ansible 集成**: 通过 playbook 自动化在整个集群执行

#### 检测原理

基于 **Microsoft Azure** 在 DGX Cloud 中使用的方法论：

```
1. 运行多次 NCCL all-reduce 测试（默认 10 次）收集统计数据
2. 当总带宽偏离基线时，使用二分搜索隔离性能不佳的节点
3. 执行成对 NCCL 测试识别坏节点
4. 分析哪些节点在慢节点对中出现频率最高
```

#### 性能基线

**节点内部（8 GPUs with NVLink）**:

| GPU 型号 | NVLink 单 GPU 带宽 | NCCL AllReduce Bus BW | 阈值 (90%) |
|---------|------------------|---------------------|-----------|
| A100 SXM4 | 600 GB/s | ~250 GB/s | 225 GB/s |
| H100 SXM5 | 900 GB/s | ~350 GB/s | 315 GB/s |
| V100 SXM2 | 300 GB/s | ~180 GB/s | 162 GB/s |

**跨节点（InfiniBand）**:

| GPU + 网络 | NCCL Bus BW | 阈值 (92%) |
|-----------|------------|----------|
| A100 + IB HDR 200Gb | ~180 GB/s | 165 GB/s |
| A100/H100 + IB NDR 400Gb | ~360 GB/s | 331 GB/s |

#### 使用方式

**单节点内部检查**:
```bash
# 快速检查单个节点的 GPU 带宽
./scripts/validation/intra_node_bandwidth_check.sh -o results

# 自定义阈值（85%）
./scripts/validation/intra_node_bandwidth_check.sh -o results -t 85
```

**跨节点 NCCL 检查**:
```bash
# 创建节点列表
cat > nodes.txt <<EOF
gpu-node1
gpu-node2
gpu-node3
gpu-node4
EOF

# 基本检查（全节点 all-reduce）
./scripts/validation/inter_node_nccl_check.sh -n nodes.txt -o results

# 启用成对测试（检测所有节点对）
./scripts/validation/inter_node_nccl_check.sh -n nodes.txt -o results --pairwise

# 启用二分搜索（快速定位慢节点）
./scripts/validation/inter_node_nccl_check.sh -n nodes.txt -o results --binary-search

# 完整检测（成对 + 二分搜索 + 20 次迭代）
./scripts/validation/inter_node_nccl_check.sh -n nodes.txt -o results \
  --pairwise --binary-search -i 20
```

**综合检测（推荐）**:
```bash
# 完整集群检测（节点内部 + 跨节点）
./scripts/validation/detect_slow_nodes.sh -n nodes.txt -o results

# 并行执行节点内部检查 + 完整跨节点测试
./scripts/validation/detect_slow_nodes.sh -n nodes.txt -o results \
  --parallel --pairwise --binary-search

# 仅节点内部检查（快速）
./scripts/validation/detect_slow_nodes.sh -n nodes.txt --skip-inter --parallel

# 仅跨节点检查
./scripts/validation/detect_slow_nodes.sh -n nodes.txt --skip-intra --pairwise
```

**Ansible 自动化**:
```bash
# 使用 Ansible 在整个集群运行检测
cd ansible

# 完整检测
ansible-playbook -i inventory playbooks/detect_slow_nodes.yml

# 自定义配置
ansible-playbook -i inventory playbooks/detect_slow_nodes.yml \
  -e slow_node_detection_threshold=92 \
  -e slow_node_detection_pairwise=true \
  -e slow_node_detection_binary_search=true \
  -e slow_node_detection_parallel=true
```

#### 检测输出

检测完成后会生成详细报告：

```
results/
├── intra_node_results/           # 节点内部检查结果
│   ├── node1_<timestamp>/
│   │   ├── gpu_info_*.txt
│   │   ├── nvlink_topology_*.txt
│   │   ├── p2p_bandwidth_summary_*.csv
│   │   ├── pcie_bandwidth_summary_*.csv
│   │   ├── slow_connections_*.txt    # ⚠ 慢 GPU 连接（如有）
│   │   └── bandwidth_check_report_*.md
│   └── ...
├── inter_node_results/           # 跨节点 NCCL 检查结果
│   ├── all_nodes_*_stats.txt
│   ├── pairwise_results_*.csv
│   └── nccl_check_report_*.md
└── slow_node_summary_*.md        # 综合报告
```

#### 常见问题诊断

**问题类型 1: NVLink 带宽低**
```
症状: GPU 0 <-> GPU 1: 150 GB/s (预期: 300 GB/s)
原因: NVLink cable 松动或故障
解决: 重新插拔 cable，检查 nvidia-smi nvlink --status
```

**问题类型 2: PCIe 降速**
```
症状: GPU 运行在 Gen3 x8 (预期: Gen4 x16)
原因: PCIe 槽位配置错误
解决: 确认 GPU 安装在正确的 PCIe 槽位，更新 BIOS
```

**问题类型 3: 跨节点通讯慢**
```
症状: node1 <-> node3 的所有成对测试均 <100 GB/s
原因: InfiniBand 连接问题或网卡故障
解决: 检查 IB cable，验证 ibstat 输出，更新 IB 驱动
```

**详细文档**: [慢节点检测完整指南](docs/slow_node_detection.md)

#### 最佳实践

**定期检测频率**:
- **新集群部署**: 立即运行完整检测作为验收
- **日常运行**: 每周快速检测（仅节点内部）
- **定期维护**: 每月完整检测（包括跨节点）
- **问题排查**: 发现性能问题时立即运行

**检测策略**:
- **快速检测** (5-10 分钟): 仅节点内部，并行执行
- **标准检测** (30-60 分钟): 节点内部 + 全节点 NCCL
- **深度检测** (2-4 小时): 完整成对测试 + 二分搜索

**自动化**:
```bash
# 使用 cron 定期运行
# /etc/cron.weekly/gpu_cluster_check
ansible-playbook -i production_inventory playbooks/detect_slow_nodes.yml \
  -e slow_node_detection_skip_inter=true \
  -e slow_node_detection_output_dir=/var/log/slow_node_detection/$(date +%Y%m%d)
```

### 9. 🆕 RDMA 环境验证 (rdma_check.sh)

**全面检查 RDMA/InfiniBand 环境是否准备就绪**，用于高性能集群通讯：

#### 检查项目

**1. RDMA 内核模块**
- ✅ **核心 RDMA 模块**: rdma_cm, ib_core, ib_uverbs, rdma_ucm
- ✅ **传输层模块**: ib_ipoib, ib_srp, ib_iser 等
- ✅ **厂商驱动**: mlx5_core, mlx4_core (Mellanox/NVIDIA)
- ✅ **GPUDirect RDMA**: nv_peer_mem / nvidia_peermem 模块

**2. RDMA 设备状态**
- ✅ **InfiniBand 设备**: ibstat 检测设备和端口状态
- ✅ **端口状态**: Active/Down 状态，链路速度 (FDR/EDR/HDR/NDR)
- ✅ **RDMA 设备信息**: ibv_devinfo 检测设备能力
- ✅ **链路层**: InfiniBand 或 RoCE (Ethernet)

**3. 软件栈完整性**
- ✅ **核心库**: libibverbs, librdmacm, rdma-core
- ✅ **诊断工具**: infiniband-diags (ibstat, ibv_devinfo 等)
- ✅ **性能测试**: perftest (ib_write_bw, ib_read_bw 等)
- ✅ **Subnet Manager**: opensm (InfiniBand 必需)

**4. 网络配置**
- ✅ **IPoIB 接口**: IP over InfiniBand 网络接口
- ✅ **MTU 配置**: Connected mode (65520) vs Datagram mode (2044)
- ✅ **RoCE 支持**: RDMA over Converged Ethernet 检测
- ✅ **接口状态**: UP/DOWN, IP 地址配置

**5. GPUDirect RDMA 环境**
- ✅ **GPU 检测**: NVIDIA GPU 和驱动版本
- ✅ **Peer Memory**: nv_peer_mem 模块和 /sys/kernel/mm/memory_peer_target
- ✅ **NUMA 亲和性**: GPU 和 IB 设备的 NUMA 节点分布
- ✅ **GPU 拓扑**: NVLink 和 PCIe 拓扑结构

**6. 系统配置**
- ✅ **IOMMU**: VT-d/AMD-Vi 启用状态和内核参数
- ✅ **内存锁定**: ulimit memlock 配置 (应为 unlimited)
- ✅ **PCIe 状态**: GPU 和 IB 设备的 PCIe 链路速度/宽度

#### 使用方式

```bash
# 运行 RDMA 环境检查
sudo ./scripts/validation/rdma_check.sh

# 指定输出目录
sudo ./scripts/validation/rdma_check.sh /path/to/output_dir
```

#### 输出报告

脚本会生成以下文件：

```
output_dir/
├── rdma_check.json           # JSON 格式详细报告
├── rdma_summary.md           # Markdown 格式摘要报告
├── ibstat_output.txt         # ibstat 原始输出
├── ibv_devinfo_output.txt    # ibv_devinfo 原始输出
├── rdma_link_output.txt      # rdma link 输出
└── gpu_topology.txt          # GPU 拓扑信息
```

#### 报告示例

```
========================================
RDMA 环境验证
========================================
开始时间: 2025-01-15 10:30:00

1. RDMA 内核模块检查
========================================
✓ [内核模块] rdma_cm: 已加载
✓ [内核模块] ib_core: 已加载
✓ [内核模块] ib_uverbs: 已加载
✓ [厂商驱动] mlx5_core: 已加载
✓ [厂商驱动] mlx5_ib: 已加载
✓ [GPUDirect] nv_peer_mem: 已加载 (版本: 1.3-0)

2. RDMA 设备检查
========================================
✓ [IB设备] 设备数量: 2
✓ [IB端口] mlx5_0:1: Active @ 200 Gb/sec (4X HDR)
✓ [IB端口] mlx5_1:1: Active @ 200 Gb/sec (4X HDR)

检查结果摘要
========================================
  ✓ 通过:  45
  ⚠ 警告:  3
  ✗ 失败:  0

========================================
✅ RDMA 环境基本就绪
========================================
```

#### 常见问题修复

**问题 1: 内核模块未加载**
```bash
# 加载 RDMA 核心模块
sudo modprobe rdma_cm
sudo modprobe ib_core
sudo modprobe ib_uverbs

# 加载 Mellanox 驱动
sudo modprobe mlx5_core
sudo modprobe mlx5_ib

# 加载 GPUDirect RDMA
sudo modprobe nv_peer_mem
```

**问题 2: 内存锁定限制**
```bash
# 编辑 /etc/security/limits.conf
sudo tee -a /etc/security/limits.conf << EOF
* soft memlock unlimited
* hard memlock unlimited
EOF

# 重新登录生效
```

**问题 3: IB 端口 Down**
```bash
# 检查物理连接
ibstat

# 检查链路状态
ibv_devinfo

# 重启 IB 驱动
sudo /etc/init.d/openibd restart
```

**问题 4: GPUDirect RDMA 不可用**
```bash
# 安装 nvidia-peer-memory (CUDA 11.x+)
# Ubuntu/Debian:
git clone https://github.com/Mellanox/nv_peer_memory.git
cd nv_peer_memory
./build_module.sh
sudo ./install.sh

# 或使用 MLNX_OFED 自带的版本
sudo /etc/init.d/nv_peer_mem start
```

#### RDMA 性能测试

检查通过后，可以进行 RDMA 性能测试：

```bash
# 测试 RDMA 写带宽 (需要两台主机)
# 服务端
ib_write_bw -d mlx5_0 -a

# 客户端
ib_write_bw -d mlx5_0 -a <server_ip>

# 测试 GPUDirect RDMA (如果支持)
# 服务端
ib_write_bw -d mlx5_0 --use_cuda=0

# 客户端
ib_write_bw -d mlx5_0 --use_cuda=0 <server_ip>

# 预期性能 (InfiniBand HDR 200Gb/s)
# - 主机内存: ~23-24 GB/s
# - GPU 内存 (GPUDirect): ~20-22 GB/s
```

#### 参考资源

- [NVIDIA GPUDirect RDMA 官方文档](https://docs.nvidia.com/cuda/gpudirect-rdma/)
- [Mellanox OFED 用户手册](https://docs.nvidia.com/networking/display/mlnxofedv24010331)
- [Linux RDMA 核心文档](https://github.com/linux-rdma/rdma-core)
- [InfiniBand 性能调优](https://docs.nvidia.com/networking/display/perftuning)

### 10. GPU 验证测试 (多级别)

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

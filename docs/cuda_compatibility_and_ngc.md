# CUDA 兼容性和 NGC 镜像指南

本文档介绍 GPU 型号与 CUDA 版本的对应关系，以及 NVIDIA NGC 容器镜像的使用。

## 目录

1. [GPU 与 CUDA 版本兼容性](#gpu-与-cuda-版本兼容性)
2. [NGC 容器镜像](#ngc-容器镜像)
3. [自动化部署](#自动化部署)
4. [使用示例](#使用示例)

---

## GPU 与 CUDA 版本兼容性

### 兼容性矩阵

#### Volta 架构 (Compute Capability 7.0)

| GPU 型号 | 最低 CUDA | 推荐 CUDA | 最低驱动 | 推荐驱动 |
|---------|-----------|-----------|----------|----------|
| Tesla V100 SXM2 | 9.0 | 12.2 | 396.26 | 535.154.05 |
| Tesla V100 PCIe | 9.0 | 12.2 | 396.26 | 535.154.05 |

**支持的 CUDA 版本**: 11.8, 12.0, 12.1, 12.2, 12.3, 12.4

#### Ampere 架构 (Compute Capability 8.0/8.6)

| GPU 型号 | 最低 CUDA | 推荐 CUDA | 最低驱动 | 推荐驱动 |
|---------|-----------|-----------|----------|----------|
| A100 SXM4 80GB | 11.0 | 12.2 | 450.51.06 | 535.154.05 |
| A100 PCIe 40GB | 11.0 | 12.2 | 450.51.06 | 535.154.05 |
| A800 | 11.0 | 12.2 | 450.51.06 | 535.154.05 |
| RTX 3090 | 11.1 | 12.2 | 455.32.00 | 535.154.05 |

**支持的 CUDA 版本**: 11.8, 12.0, 12.1, 12.2, 12.3, 12.4

#### Hopper 架构 (Compute Capability 9.0)

| GPU 型号 | 最低 CUDA | 推荐 CUDA | 最低驱动 | 推荐驱动 |
|---------|-----------|-----------|----------|----------|
| H100 SXM5 | 11.8 | 12.3 | 520.61.05 | 545.23.08 |
| H100 PCIe | 11.8 | 12.3 | 520.61.05 | 545.23.08 |
| H800 | 11.8 | 12.3 | 520.61.05 | 545.23.08 |

**支持的 CUDA 版本**: 11.8, 12.0, 12.1, 12.2, 12.3, 12.4

**注意**: H100 需要 CUDA 11.8+ 以支持全部特性（包括 FP8）

#### Ada Lovelace 架构 (Compute Capability 8.9)

| GPU 型号 | 最低 CUDA | 推荐 CUDA | 最低驱动 | 推荐驱动 |
|---------|-----------|-----------|----------|----------|
| RTX 4090 | 11.8 | 12.2 | 520.61.05 | 535.154.05 |

**支持的 CUDA 版本**: 11.8, 12.0, 12.1, 12.2, 12.3, 12.4

### CUDA 版本到驱动版本映射

| CUDA 版本 | 最低驱动 (Linux) | 推荐驱动 (Linux) |
|-----------|-----------------|-----------------|
| 12.4 | 550.54.15 | 550.90.07 |
| 12.3 | 545.23.06 | 545.23.08 |
| 12.2 | 535.54.03 | 535.154.05 |
| 12.1 | 530.30.02 | 530.30.02 |
| 12.0 | 525.60.13 | 525.125.06 |
| 11.8 | 520.61.05 | 520.61.05 |

### 查询 GPU 兼容性

使用 `cuda_compatibility.py` 脚本查询 GPU 兼容的 CUDA 版本：

```bash
# 查看完整兼容性矩阵
python3 scripts/utils/cuda_compatibility.py --matrix

# 查询特定 GPU
python3 scripts/utils/cuda_compatibility.py "A100-SXM4"

# 输出示例:
# GPU: A100-SXM4
# Architecture: Ampere
# Compute Capability: 8.0
# Recommended CUDA: 12.2
# Recommended Driver: 535.154.05
# Supported CUDA Versions: 11.8, 12.0, 12.1, 12.2, 12.3, 12.4
```

---

## NGC 容器镜像

NVIDIA NGC (GPU Cloud) 提供了预配置、优化的容器镜像，包含深度学习框架和推理服务器。

### 可用镜像列表

#### 1. PyTorch

**镜像**: `nvcr.io/nvidia/pytorch:24.01-py3`

**特性**:
- PyTorch 2.3.0a0
- CUDA 12.3
- cuDNN 8.9
- NCCL 2.19
- TensorRT 8.6
- Python 3.10

**用途**: 训练、推理、开发

#### 2. NeMo (包含 Megatron-LM)

**镜像**: `nvcr.io/nvidia/nemo:24.01`

**特性**:
- NeMo 1.22.0
- Megatron-LM Core 0.5.0
- PyTorch 2.2.0
- CUDA 12.3
- Transformer Engine
- Apex

**用途**: LLM 训练、LLM 微调、ASR、TTS、NLP

#### 3. Triton Inference Server

**镜像**: `nvcr.io/nvidia/tritonserver:24.01-py3`

**特性**:
- Triton 2.42.0
- CUDA 12.3
- TensorRT 8.6
- PyTorch Backend
- TensorFlow Backend
- ONNX Runtime
- Python Backend

**用途**: 模型推理、生产部署、模型服务

#### 4. TensorFlow

**镜像**: `nvcr.io/nvidia/tensorflow:24.01-tf2-py3`

**特性**:
- TensorFlow 2.15.0
- CUDA 12.3
- cuDNN 8.9
- NCCL 2.19

**用途**: 训练、推理

#### 5. TensorRT

**镜像**: `nvcr.io/nvidia/tensorrt:24.01-py3`

**特性**:
- TensorRT 8.6.3
- CUDA 12.3
- cuDNN 8.9
- ONNX Parser

**用途**: 推理优化、模型转换

#### 6. CUDA 开发环境

**镜像**: `nvcr.io/nvidia/cuda:12.3.2-devel-ubuntu22.04`

**特性**:
- CUDA Toolkit 12.3.2
- NVCC 编译器
- cuBLAS, cuFFT, cuSPARSE

**用途**: CUDA 开发、自定义应用

#### 7. RAPIDS

**镜像**: `nvcr.io/nvidia/rapidsai/rapids:24.02-cuda12.0-py3.10`

**特性**:
- RAPIDS 24.02
- cuDF, cuML, cuGraph
- Dask 支持

**用途**: 数据科学、机器学习预处理、ETL

#### 8. DeepStream

**镜像**: `nvcr.io/nvidia/deepstream:6.4-triton-multiarch`

**特性**:
- DeepStream 6.4
- Triton 集成
- 视频分析

**用途**: 视频分析、流式 AI

### NGC 镜像版本选择

查看特定 CUDA 版本兼容的镜像：

```bash
python3 scripts/utils/ngc_images.py --cuda 12.3

# 输出示例:
# NGC Images compatible with CUDA 12.3:
#   PyTorch: nvcr.io/nvidia/pytorch:24.01-py3
#   NeMo: nvcr.io/nvidia/nemo:24.01
#   Triton Inference Server: nvcr.io/nvidia/tritonserver:24.01-py3
```

查看镜像详细信息：

```bash
python3 scripts/utils/ngc_images.py nemo

# 输出包含:
# - 版本信息
# - CUDA 版本
# - 特性列表
# - 使用场景
```

---

## 自动化部署

### 1. GPU 自动检测和 CUDA 版本选择

Ansible `gpu_baseline` role 支持自动检测 GPU 型号并选择对应的 CUDA 版本：

**配置文件**: `ansible/roles/gpu_baseline/defaults/main.yml`

```yaml
# 启用自动检测
auto_detect_cuda_version: true

# 默认版本（自动检测失败时使用）
nvidia_driver_version: "535"
cuda_version: "12-2"
```

**运行 playbook**:

```bash
cd ansible
ansible-playbook -i inventory/hosts playbooks/setup_gpu_baseline.yml

# 输出会显示:
# - 检测到的 GPU 型号
# - 自动选择的 CUDA 版本
# - 自动选择的驱动版本
```

**检测日志**: `/var/log/gpu_baseline/gpu_detection.txt`

### 2. NGC 镜像管理

使用 `ngc_images` role 自动拉取和测试 NGC 镜像：

**配置文件**: `ansible/roles/ngc_images/defaults/main.yml`

```yaml
# 指定要拉取的镜像
ngc_images_to_pull:
  - name: pytorch
    version: "24.01"
  - name: nemo
    version: "24.01"
  - name: triton
    version: "24.01"

# 自动根据 CUDA 版本选择镜像
auto_select_images_by_cuda: true

# 拉取后测试镜像
test_images_after_pull: true
```

**运行 playbook**:

```bash
cd ansible
ansible-playbook -i inventory/hosts playbooks/setup_ngc_images.yml
```

**查看报告**: `/var/log/ngc_images/ngc_inventory_report.txt`

---

## 使用示例

### 示例 1: 使用 NGC 管理脚本

```bash
# 列出可用镜像
./scripts/utils/ngc_manager.sh list

# 拉取 PyTorch 镜像
./scripts/utils/ngc_manager.sh pull pytorch

# 拉取特定版本
./scripts/utils/ngc_manager.sh pull pytorch 24.01

# 运行镜像（交互式）
./scripts/utils/ngc_manager.sh run pytorch

# 测试镜像 GPU 功能
./scripts/utils/ngc_manager.sh test pytorch

# 查看镜像信息
./scripts/utils/ngc_manager.sh info nemo

# 查看 CUDA 12.3 兼容镜像
./scripts/utils/ngc_manager.sh cuda 12.3
```

### 示例 2: 使用 NGC PyTorch 镜像训练

```bash
# 拉取镜像
docker pull nvcr.io/nvidia/pytorch:24.01-py3

# 运行容器
docker run --gpus all -it --rm \
  --ipc=host \
  --network=host \
  -v $HOME/workspace:/workspace \
  nvcr.io/nvidia/pytorch:24.01-py3

# 在容器中
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
```

### 示例 3: 使用 NGC NeMo 镜像进行 Megatron 训练

```bash
# 使用 NGC 容器运行 Megatron benchmark
export USE_NGC_CONTAINER=true
export NGC_IMAGE=nvcr.io/nvidia/nemo:24.01

./scripts/benchmarks/megatron_benchmark.sh

# 或者指定不同的镜像版本
NGC_IMAGE=nvcr.io/nvidia/nemo:23.11 \
  ./scripts/benchmarks/megatron_benchmark.sh
```

### 示例 4: 使用 Triton Inference Server

```bash
# 拉取 Triton 镜像
docker pull nvcr.io/nvidia/tritonserver:24.01-py3

# 准备模型仓库
mkdir -p /tmp/model_repository

# 运行 Triton Server
docker run --gpus all -it --rm \
  --shm-size=1g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v /tmp/model_repository:/models \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models
```

### 示例 5: 自动选择最佳配置

```bash
# 检测 GPU 并获取推荐配置
GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "Detected GPU: $GPU_MODEL"

# 获取推荐 CUDA 版本
CUDA_VER=$(python3 scripts/utils/cuda_compatibility.py "$GPU_MODEL" 2>/dev/null | \
  grep "Recommended CUDA:" | awk '{print $3}')
echo "Recommended CUDA: $CUDA_VER"

# 获取兼容的 NGC 镜像
python3 scripts/utils/ngc_images.py --cuda "$CUDA_VER"

# 自动拉取推荐镜像
./scripts/utils/ngc_manager.sh pull pytorch
./scripts/utils/ngc_manager.sh pull nemo
```

### 示例 6: 批量部署

创建 playbook: `playbooks/deploy_full_stack.yml`

```yaml
---
- name: Deploy GPU Full Stack
  hosts: gpu_nodes
  become: yes
  roles:
    # 1. 检测 GPU 并安装对应 CUDA
    - role: gpu_baseline
      vars:
        auto_detect_cuda_version: true

    # 2. CPU 性能优化
    - role: cpu_optimization

    # 3. 拉取 NGC 镜像
    - role: ngc_images
      vars:
        auto_select_images_by_cuda: true
        ngc_images_to_pull:
          - name: pytorch
          - name: nemo
          - name: triton

    # 4. 安装 benchmark 工具
    - role: benchmark_tools
```

运行：

```bash
cd ansible
ansible-playbook -i inventory/hosts playbooks/deploy_full_stack.yml
```

---

## 最佳实践

### 1. CUDA 版本选择

- **V100**: 使用 CUDA 12.2（平衡兼容性和性能）
- **A100**: 使用 CUDA 12.2（最佳兼容性）
- **H100**: 使用 CUDA 12.3+（支持 FP8 等新特性）
- **RTX 4090**: 使用 CUDA 12.2

### 2. NGC 镜像选择

- **训练**: 优先使用 PyTorch 或 NeMo 镜像
- **推理**: 使用 Triton Server 或 TensorRT 镜像
- **LLM**: 使用 NeMo 镜像（包含 Megatron-LM）
- **开发**: 使用 CUDA 开发镜像

### 3. 驱动更新

- 优先使用推荐驱动版本
- 定期检查 NVIDIA 驱动更新
- 测试环境先升级，生产环境后升级

### 4. 容器最佳实践

- 使用 `--ipc=host` 以获得更好的共享内存性能
- 使用 `--network=host` 简化多节点通信
- 挂载必要的数据卷（`-v`）
- 限制 GPU 可见性（`CUDA_VISIBLE_DEVICES`）

---

## 故障排除

### 问题 1: CUDA 版本不兼容

**症状**: 程序报错 "CUDA version mismatch" 或 "Unsupported GPU architecture"

**解决**:
```bash
# 检查 CUDA 版本
nvcc --version

# 检查驱动版本
nvidia-smi

# 查看 GPU 支持的 CUDA 版本
python3 scripts/utils/cuda_compatibility.py "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# 使用匹配的 NGC 镜像
./scripts/utils/ngc_manager.sh cuda <your_cuda_version>
```

### 问题 2: NGC 镜像拉取失败

**症状**: Docker pull 超时或失败

**解决**:
```bash
# 检查网络连接
ping nvcr.io

# 增加 docker pull 超时
export DOCKER_CLIENT_TIMEOUT=600
export COMPOSE_HTTP_TIMEOUT=600

# 使用镜像代理（如有）
# 配置 /etc/docker/daemon.json

# 手动拉取
docker pull nvcr.io/nvidia/pytorch:24.01-py3
```

### 问题 3: 容器无法访问 GPU

**症状**: `torch.cuda.is_available()` 返回 False

**解决**:
```bash
# 检查 nvidia-container-toolkit
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi

# 重启 docker
sudo systemctl restart docker

# 检查 GPU 可见性
nvidia-smi

# 使用正确的 GPU 标志
docker run --gpus all ...  # 使用所有 GPU
docker run --gpus '"device=0,1"' ...  # 使用特定 GPU
```

---

## 参考资料

- [NVIDIA CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/)
- [NGC Container Registry](https://catalog.ngc.nvidia.com/)
- [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
- [PyTorch NGC Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
- [NeMo Framework](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo)
- [Triton Inference Server](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver)

# GPU Passthrough - 基线安装与验证自动化

这个项目提供基于 Ansible 的 GPU 机器基线安装和验证自动化解决方案，基于 NVIDIA 官方工具和开源社区最佳实践。

## 项目目标

1. **自动化安装**: 通过 Ansible 自动化安装 GPU 机器的基线环境（驱动、CUDA、容器运行时）
2. **验证测试**: 提供多级别的验证脚本，确保机器可以正常交付和使用
3. **开源整合**: 基于 NVIDIA 官方和社区开源项目的最佳实践

## 项目结构

```
gpu_passthrough/
├── ansible/                    # Ansible 自动化配置
│   ├── roles/
│   │   ├── gpu_baseline/      # GPU 基线安装 role
│   │   └── gpu_validation/    # GPU 验证 role
│   ├── playbooks/             # Playbook 文件
│   ├── inventory/             # 主机清单
│   └── ansible.cfg
├── scripts/                    # 验证和监控脚本
│   ├── validation/            # 验证脚本
│   ├── monitoring/            # 监控脚本
│   └── utils/                 # 工具脚本
├── docs/                       # 文档
│   ├── research.md            # 开源项目调研报告
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

### 2. GPU 验证测试 (多级别)

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
nvidia_driver_version: "535"
cuda_version: "12-2"
install_cuda: true
install_container_runtime: true
container_runtime: "docker"
```

#### 4. 运行基线安装

```bash
cd ansible
ansible-playbook playbooks/setup_gpu_baseline.yml
```

#### 5. 运行验证

```bash
# 快速验证
ansible-playbook playbooks/validate_gpu.yml -e "level=quick"

# 标准验证
ansible-playbook playbooks/validate_gpu.yml -e "level=standard"

# 完整验证（交付前）
ansible-playbook playbooks/validate_gpu.yml -e "level=full"
```

### 单独使用验证脚本

```bash
# 快速检查
./scripts/validation/quick_check.sh /tmp/gpu_check.json

# Python 健康检查
python3 scripts/validation/gpu_health.py -o /tmp/health_report.json -v
```

## 配置说明

### 关键变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `nvidia_driver_version` | "535" | NVIDIA 驱动版本 |
| `cuda_version` | "12-2" | CUDA Toolkit 版本 |
| `install_cuda` | true | 是否安装 CUDA |
| `install_container_runtime` | true | 是否安装容器运行时 |
| `container_runtime` | "docker" | 容器运行时类型 (docker/containerd) |
| `gpu_persistence_mode` | true | GPU 持久化模式 |
| `validation_level` | "quick" | 验证级别 (quick/standard/full) |

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

## 开源项目参考

本项目整合了以下优秀的开源项目和工具：

### Ansible Roles
- [NVIDIA/ansible-role-nvidia-driver](https://github.com/NVIDIA/ansible-role-nvidia-driver)
- [NVIDIA/ansible-role-nvidia-docker](https://github.com/NVIDIA/ansible-role-nvidia-docker)
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

## 文档

- [开源项目调研报告](docs/research.md) - 详细的开源项目调研和分析
- [实施方案](docs/implementation_plan.md) - 完整的技术实施方案

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

本项目基于 MIT 许可证开源。

## 致谢

感谢 NVIDIA 和开源社区提供的优秀工具和最佳实践。

---

**项目维护者**: 请根据实际情况更新

**最后更新**: 2025-01-15

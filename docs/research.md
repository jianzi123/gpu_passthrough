# GPU 基线安装与验证 - 开源项目调研报告

## 项目目标
1. 通过 Ansible 自动化安装 GPU 机器的基线环境
2. 通过脚本验证机器可以交付并正常使用
3. 基于开源项目的最佳实践

## 一、Ansible GPU 驱动安装相关项目

### 1.1 官方 NVIDIA 项目

#### NVIDIA/ansible-role-nvidia-driver
- **仓库**: https://github.com/NVIDIA/ansible-role-nvidia-driver
- **描述**: NVIDIA 官方提供的 Ansible role，用于从 CUDA 仓库安装 NVIDIA 驱动
- **特点**:
  - 官方支持，更新及时
  - 自动处理驱动安装和系统重启
  - 建议从独立节点运行 playbook
- **使用场景**: 生产环境的标准驱动安装

#### NVIDIA/ansible-role-nvidia-docker
- **仓库**: https://github.com/NVIDIA/ansible-role-nvidia-docker
- **描述**: 安装 nvidia-docker 的官方 Ansible role
- **特点**:
  - 与 NVIDIA Container Toolkit 集成
  - 支持容器化 GPU 工作负载
- **使用场景**: 容器化环境的 GPU 支持

### 1.2 社区优秀项目

#### CSCfi/ansible-role-cuda
- **仓库**: https://github.com/fgci-org/ansible-role-cuda
- **描述**: 安装 CUDA 工具包的 Ansible role
- **特点**:
  - 支持 Ubuntu 和 RHEL/CentOS
  - 可指定 CUDA 版本
  - 配置灵活

#### Provizanta/ansible-role-nvidia-cuda
- **仓库**: https://github.com/Provizanta/ansible-role-nvidia-cuda
- **描述**: 在 Linux 上安装显卡驱动的 Ansible role
- **特点**:
  - 可选择性安装最新 CUDA 驱动
  - 支持多 Linux 发行版

#### datadrivers/ansible-role-docker
- **仓库**: https://github.com/datadrivers/ansible-role-docker
- **描述**: 部署 Docker 环境，包含 GPU 支持
- **特点**:
  - 集成 docker-compose
  - 包含 NVIDIA GPU 支持
  - 附带 ctop 等监控工具
- **使用场景**: 完整的容器化环境部署

#### CloudVE/ansible-gpu
- **仓库**: https://github.com/CloudVE/ansible-gpu
- **描述**: 安装 Nvidia CUDA 驱动、Docker 和 Nvidia Docker 运行时
- **特点**:
  - 一站式解决方案
  - 适合云环境部署

### 1.3 GPU 透传相关项目

#### capta1nk1rk/containerD_gpu_patch
- **仓库**: https://github.com/capta1nk1rk/containerD_gpu_patch
- **描述**: 用于 containerD NVIDIA GPU 透传的 Ansible Playbooks 和脚本
- **特点**:
  - 专注于 containerD
  - 包含补丁脚本

#### simoncaron/ansible-role-pve_nvidia_passthrough
- **仓库**: https://github.com/simoncaron/ansible-role-pve_nvidia_passthrough
- **描述**: 在 Proxmox VE 上配置 NVIDIA 驱动用于 GPU 透传
- **特点**:
  - 支持 VM 和 LXC 透传
  - Proxmox VE 7.x 专用
- **使用场景**: 虚拟化环境中的 GPU 透传

## 二、GPU 验证和测试工具

### 2.1 NVIDIA 官方验证工具

#### NVIDIA Validation Suite (NVVS)
- **文档**: https://docs.nvidia.com/deploy/nvvs-user-guide/
- **描述**: 系统管理员和集群管理器用于检测和排除 Tesla GPU 故障的工具
- **特点**:
  - 可脚本化运行
  - JSON 格式输出，易于解析
  - 支持快速测试和完整验证
  - 快速测试作为预运行检查
  - 完整测试时间 10-60 分钟（取决于 GPU 数量）
- **使用场景**: 生产环境的 GPU 健康检查

#### NVIDIA Data Center GPU Manager (DCGM)
- **仓库**: https://github.com/NVIDIA/DCGM
- **文档**: https://docs.nvidia.com/datacenter/dcgm/
- **描述**: 数据中心 GPU 管理器，用于收集遥测数据和测量 GPU 健康状况
- **特点**:
  - 包含诊断测试套件
  - 支持长时间侵入性健康检查
  - 检测所有 GPU 和 NVLink 交换机
  - 提供 Python 脚本用于数据可视化
  - 平均测试时间约 30 分钟
- **关键命令**:
  - `dcgmi discovery -l`: 发现所有 GPU
  - 诊断测试支持不同级别（Quick/Extended/Long）
- **使用场景**: 数据中心级别的 GPU 监控和诊断

#### NVIDIA GPU Stress Test
- **仓库**: https://github.com/NVIDIA/GPUStressTest
- **描述**: 通过运行 BLAS 矩阵乘法压测 Tesla GPU
- **特点**:
  - 支持不同数据类型
  - Linux 和 Windows 都支持
  - 包含 Docker 辅助工具
  - MPI 支持
- **使用场景**: GPU 压力测试

### 2.2 社区监控和测试工具

#### wookayin/gpustat
- **仓库**: https://github.com/wookayin/gpustat
- **描述**: 查询和监控 GPU 状态的命令行工具
- **特点**:
  - 比 nvidia-smi 更简洁的界面
  - Python 实现，易于集成
- **使用场景**: 日常 GPU 状态监控

#### XuehaiPan/nvitop
- **仓库**: https://github.com/XuehaiPan/nvitop
- **描述**: 交互式 NVIDIA GPU 进程查看器
- **特点**:
  - 直接使用 NVML Python 绑定
  - 不需要解析 nvidia-smi 输出
  - GPU 进程管理一站式解决方案
- **使用场景**: 交互式 GPU 进程监控

#### ahnilica/gpu-tester
- **仓库**: https://github.com/ahnilica/gpu-tester
- **描述**: 测试 GPU 状况的程序套件（Ubuntu）
- **特点**:
  - 包含健康检查脚本
  - Ubuntu 特定
- **使用场景**: Ubuntu 环境的 GPU 测试

#### anderskm/gputil
- **仓库**: https://github.com/anderskm/gputil
- **描述**: 使用 nvidia-smi 在 Python 中获取 GPU 状态
- **特点**:
  - 编程方式访问 GPU 状态
  - 易于集成到自动化脚本
- **使用场景**: Python 自动化脚本

### 2.3 压力测试工具

#### GPU-Burn
- **描述**: GPU 压力测试工具
- **特点**:
  - 长时间运行压力测试
  - 检测内存错误和其他故障
  - 确保持续高负载下的稳定性
- **使用场景**: 烧机测试

## 三、最佳实践和实施建议

### 3.1 基线安装流程

基于调研的开源项目，建议的安装流程：

1. **操作系统基线配置**
   - 更新系统包
   - 配置内核参数
   - 禁用 nouveau 驱动

2. **NVIDIA 驱动安装**
   - 使用 NVIDIA 官方 Ansible role
   - 从 CUDA 仓库安装
   - 处理系统重启

3. **CUDA 工具包安装**
   - 可选特定 CUDA 版本
   - 配置环境变量

4. **容器运行时（可选）**
   - 安装 Docker/containerd
   - 配置 NVIDIA Container Toolkit
   - 设置默认运行时

5. **验证安装**
   - 运行基础验证脚本
   - 检查驱动加载状态

### 3.2 验证测试流程

建议的验证流程分为三个级别：

#### Level 1: 快速验证（1-5 分钟）
- `nvidia-smi` 命令检查
- GPU 可见性验证
- 驱动版本确认
- 基础信息收集

#### Level 2: 标准验证（10-15 分钟）
- DCGM Quick 诊断
- 简单计算任务测试
- 内存访问测试
- PCIe 带宽测试

#### Level 3: 完整验证（30-60 分钟）
- DCGM 完整诊断套件
- GPU-Burn 压力测试
- 长时间稳定性测试
- NVLink/GPUDirect 测试（如适用）

### 3.3 监控和日志

推荐的监控方案：

1. **实时监控**
   - DCGM 用于生产监控
   - nvitop/gpustat 用于交互式查看

2. **日志收集**
   - nvidia-smi 查询输出为 CSV 格式
   - DCGM JSON 格式输出
   - 集成到现有日志系统

3. **告警机制**
   - GPU 温度异常
   - 内存错误
   - GPU 掉线
   - 性能降级

## 四、技术栈选择建议

### 4.1 Ansible Roles 组合

**推荐组合方案 1: 官方路线**
```yaml
roles:
  - nvidia.nvidia_driver
  - nvidia.nvidia_docker
  - 自定义验证 role
```

**推荐组合方案 2: 社区集成方案**
```yaml
roles:
  - datadrivers.ansible-role-docker  # 包含 GPU 支持
  - 自定义 CUDA role
  - 自定义验证 role
```

### 4.2 验证工具组合

**基础验证工具栈**
- nvidia-smi (基础)
- gpustat (日常监控)
- 自定义健康检查脚本

**生产验证工具栈**
- DCGM (核心诊断)
- NVVS (全面验证)
- GPU-Burn (压力测试)
- 自定义验证框架

## 五、参考实现代码片段

### 5.1 nvidia-smi 自动化查询

```bash
# CSV 格式输出，便于脚本解析
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free,memory.used,temperature.gpu,utilization.gpu,utilization.memory --format=csv,noheader,nounits
```

### 5.2 DCGM 诊断命令

```bash
# 发现所有 GPU
dcgmi discovery -l

# 运行快速诊断
dcgmi diag -r 1

# 运行中等诊断
dcgmi diag -r 2

# 运行完整诊断
dcgmi diag -r 3
```

### 5.3 Docker GPU 运行时配置

```json
{
  "default-runtime": "nvidia",
  "runtimes": {
    "nvidia": {
      "path": "/usr/bin/nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
```

## 六、项目实施计划

### Phase 1: 基础框架搭建
1. 创建 Ansible 项目结构
2. 集成官方 NVIDIA roles
3. 基础配置模板

### Phase 2: 验证脚本开发
1. Level 1 快速验证脚本
2. Level 2 标准验证脚本
3. Level 3 完整验证脚本
4. 验证报告生成

### Phase 3: 集成和测试
1. 端到端流程测试
2. 不同硬件配置测试
3. 文档完善

### Phase 4: 生产化
1. 错误处理和重试机制
2. 日志和监控集成
3. CI/CD 集成

## 七、关键注意事项

1. **驱动安装需要重启**: Ansible playbook 应在独立节点运行
2. **内核参数配置**: 需要配置 IOMMU、禁用 nouveau
3. **版本兼容性**: 注意 CUDA、驱动、内核版本的兼容性矩阵
4. **测试时间规划**: 完整验证可能需要 1 小时以上
5. **容器运行时选择**: Docker vs containerd，根据实际需求选择
6. **监控数据持久化**: DCGM 和 nvidia-smi 输出应持久化存储

## 八、相关资源链接

### 官方文档
- NVIDIA Driver Installation: https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/
- DCGM User Guide: https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/
- NVVS User Guide: https://docs.nvidia.com/deploy/nvvs-user-guide/

### GitHub 仓库
- NVIDIA Ansible Roles: https://galaxy.ansible.com/nvidia
- DCGM: https://github.com/NVIDIA/DCGM
- GPU Stress Test: https://github.com/NVIDIA/GPUStressTest

### 社区资源
- nvidia-smi Cheat Sheet: https://gist.github.com/omerfsen/8ecb620675525ac724a92bdf5a31a4b3
- Ansible Galaxy: https://galaxy.ansible.com/

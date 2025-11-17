# GPU 基线安装与验证 - 实施方案

## 项目结构

```
gpu_passthrough/
├── ansible/
│   ├── inventory/
│   │   ├── hosts.yml              # 主机清单
│   │   └── group_vars/
│   │       └── gpu_nodes.yml      # GPU 节点变量
│   ├── roles/
│   │   ├── gpu_baseline/          # GPU 基线安装 role
│   │   │   ├── tasks/
│   │   │   │   ├── main.yml
│   │   │   │   ├── prerequisites.yml
│   │   │   │   ├── nvidia_driver.yml
│   │   │   │   ├── cuda.yml
│   │   │   │   └── container_runtime.yml
│   │   │   ├── templates/
│   │   │   │   └── daemon.json.j2
│   │   │   ├── handlers/
│   │   │   │   └── main.yml
│   │   │   └── defaults/
│   │   │       └── main.yml
│   │   └── gpu_validation/        # GPU 验证 role
│   │       ├── tasks/
│   │       │   ├── main.yml
│   │       │   ├── level1_quick.yml
│   │       │   ├── level2_standard.yml
│   │       │   └── level3_full.yml
│   │       ├── files/
│   │       │   └── validation_scripts/
│   │       └── templates/
│   │           └── validation_report.html.j2
│   ├── playbooks/
│   │   ├── setup_gpu_baseline.yml # 安装基线
│   │   ├── validate_gpu.yml       # 验证 GPU
│   │   └── full_deployment.yml    # 完整部署
│   └── ansible.cfg
├── scripts/
│   ├── validation/
│   │   ├── quick_check.sh         # Level 1 快速检查
│   │   ├── standard_check.sh      # Level 2 标准检查
│   │   ├── full_validation.sh     # Level 3 完整验证
│   │   └── gpu_health.py          # GPU 健康检查 Python 脚本
│   ├── monitoring/
│   │   ├── gpu_monitor.sh         # GPU 监控脚本
│   │   └── collect_metrics.py     # 指标收集
│   └── utils/
│       ├── parse_nvidia_smi.py    # 解析 nvidia-smi 输出
│       └── report_generator.py    # 生成验证报告
├── docs/
│   ├── research.md                # 调研报告
│   ├── implementation_plan.md     # 实施方案（本文档）
│   └── user_guide.md             # 使用指南
├── tests/
│   └── test_validation.py         # 验证脚本的测试
└── README.md
```

## Ansible Role 设计

### 1. gpu_baseline Role

**功能**: 安装 GPU 基线环境

**主要任务**:

#### 1.1 prerequisites.yml
```yaml
- 更新系统包
- 安装依赖包 (gcc, make, kernel-headers)
- 配置内核参数 (IOMMU, intel_iommu)
- 禁用 nouveau 驱动
- 配置模块黑名单
- 更新 initramfs
```

#### 1.2 nvidia_driver.yml
```yaml
- 添加 NVIDIA CUDA 仓库
- 安装 NVIDIA 驱动
- 加载 nvidia 模块
- 验证驱动安装
- 处理重启（如需要）
```

#### 1.3 cuda.yml
```yaml
- 安装 CUDA Toolkit（可选版本）
- 配置环境变量
- 创建符号链接
- 验证 CUDA 安装
```

#### 1.4 container_runtime.yml
```yaml
- 安装 Docker/containerd
- 安装 NVIDIA Container Toolkit
- 配置容器运行时
- 设置默认运行时为 nvidia
- 重启容器服务
- 测试容器 GPU 访问
```

**关键变量** (defaults/main.yml):
```yaml
nvidia_driver_version: "latest"
cuda_version: "12.2"
install_container_runtime: true
container_runtime: "docker"  # docker or containerd
reboot_timeout: 600
```

### 2. gpu_validation Role

**功能**: 验证 GPU 功能和性能

**验证级别**:

#### 2.1 Level 1 - 快速验证 (1-5 分钟)
```yaml
检查项:
- nvidia-smi 命令可用性
- GPU 设备可见性
- 驱动版本
- CUDA 版本
- GPU 基本信息 (型号、内存)
- 温度和功耗
- PCIe 链路状态
```

#### 2.2 Level 2 - 标准验证 (10-15 分钟)
```yaml
检查项:
- Level 1 所有项
- DCGM 快速诊断
- 简单 CUDA 程序运行
- 内存带宽测试
- GPU 间通信测试（多 GPU）
- 容器 GPU 访问测试
- 错误计数检查
```

#### 2.3 Level 3 - 完整验证 (30-60 分钟)
```yaml
检查项:
- Level 2 所有项
- DCGM 完整诊断套件
- GPU-Burn 压力测试 (15-30 分钟)
- 长时间稳定性测试
- NVLink 带宽测试（如适用）
- ECC 内存测试（如支持）
- 性能基准测试
- 生成详细报告
```

**输出**:
- JSON 格式验证结果
- HTML 验证报告
- 问题汇总和建议

## 验证脚本设计

### 1. quick_check.sh

**目的**: 快速检查 GPU 基本可用性

**核心功能**:
```bash
#!/bin/bash
# 检查 nvidia-smi
# 检查 GPU 数量
# 检查驱动版本
# 检查 GPU 状态
# 输出 JSON 结果
```

**输出格式**:
```json
{
  "status": "pass|fail",
  "timestamp": "2025-01-15T10:30:00Z",
  "checks": {
    "nvidia_smi": "pass",
    "gpu_count": 8,
    "driver_version": "535.104.05",
    "gpus": [...]
  }
}
```

### 2. standard_check.sh

**目的**: 标准验证流程

**核心功能**:
```bash
#!/bin/bash
# 运行 quick_check
# 运行 DCGM 诊断
# 运行简单计算测试
# 检查错误计数
# 测试容器 GPU 访问
# 生成报告
```

### 3. full_validation.sh

**目的**: 完整验证和压力测试

**核心功能**:
```bash
#!/bin/bash
# 运行 standard_check
# 运行 GPU-Burn
# 长时间监控
# 性能基准测试
# 生成详细报告
```

### 4. gpu_health.py

**目的**: Python 实现的健康检查工具

**核心功能**:
```python
- 使用 pynvml 库直接查询 GPU
- 收集详细指标
- 执行自定义测试
- 生成结构化报告
- 支持持续监控模式
```

**主要模块**:
```python
class GPUHealthChecker:
    def check_driver()
    def check_gpus()
    def check_temperature()
    def check_memory()
    def check_processes()
    def run_compute_test()
    def generate_report()
```

## Playbook 设计

### 1. setup_gpu_baseline.yml

**目的**: 安装 GPU 基线环境

```yaml
---
- name: Setup GPU Baseline
  hosts: gpu_nodes
  become: yes
  roles:
    - gpu_baseline

  pre_tasks:
    - name: Check system requirements
      ...

  post_tasks:
    - name: Verify installation
      ...
```

### 2. validate_gpu.yml

**目的**: 验证 GPU 功能

```yaml
---
- name: Validate GPU
  hosts: gpu_nodes
  become: yes
  vars:
    validation_level: "{{ level | default('quick') }}"

  roles:
    - role: gpu_validation
      vars:
        level: "{{ validation_level }}"
```

**使用方式**:
```bash
# 快速验证
ansible-playbook validate_gpu.yml -e "level=quick"

# 标准验证
ansible-playbook validate_gpu.yml -e "level=standard"

# 完整验证
ansible-playbook validate_gpu.yml -e "level=full"
```

### 3. full_deployment.yml

**目的**: 完整部署流程

```yaml
---
- name: Full GPU Deployment
  hosts: gpu_nodes
  become: yes

  tasks:
    - name: Setup baseline
      import_role:
        name: gpu_baseline

    - name: Run validation
      import_role:
        name: gpu_validation
      vars:
        level: standard

    - name: Generate deployment report
      ...
```

## 核心技术实现

### 1. 基于开源项目的集成

#### 驱动安装
```yaml
# 使用 NVIDIA 官方 role
- name: Install NVIDIA driver
  include_role:
    name: nvidia.nvidia_driver
  vars:
    nvidia_driver_ubuntu_install_from_cuda_repo: true
```

#### DCGM 集成
```bash
# 安装 DCGM
apt-get install -y datacenter-gpu-manager

# 运行诊断
dcgmi diag -r 1  # Quick
dcgmi diag -r 2  # Extended
dcgmi diag -r 3  # Long
```

#### GPU-Burn 集成
```bash
# 克隆和编译
git clone https://github.com/wilicc/gpu-burn
cd gpu-burn
make

# 运行压力测试
./gpu_burn 900  # 15 分钟测试
```

### 2. 监控指标收集

#### nvidia-smi 查询
```bash
# 收集关键指标
nvidia-smi --query-gpu=timestamp,name,index,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,power.draw,pcie.link.gen.current,pcie.link.width.current --format=csv -l 5 >> gpu_metrics.csv
```

#### Python 实现
```python
import pynvml

pynvml.nvmlInit()
device_count = pynvml.nvmlDeviceGetCount()

for i in range(device_count):
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    temp = pynvml.nvmlDeviceGetTemperature(handle, 0)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    # 收集更多指标...
```

### 3. 报告生成

#### HTML 报告模板
```jinja2
<!DOCTYPE html>
<html>
<head>
    <title>GPU Validation Report</title>
</head>
<body>
    <h1>GPU Validation Report</h1>
    <h2>Summary</h2>
    <p>Status: {{ status }}</p>
    <p>Validation Level: {{ level }}</p>

    <h2>GPU Details</h2>
    {% for gpu in gpus %}
    <div class="gpu">
        <h3>GPU {{ gpu.index }}: {{ gpu.name }}</h3>
        <ul>
            <li>Temperature: {{ gpu.temperature }}°C</li>
            <li>Memory: {{ gpu.memory_used }}/{{ gpu.memory_total }} MB</li>
        </ul>
    </div>
    {% endfor %}
</body>
</html>
```

## 使用流程

### 初始部署

```bash
# 1. 配置主机清单
vim ansible/inventory/hosts.yml

# 2. 配置变量
vim ansible/inventory/group_vars/gpu_nodes.yml

# 3. 运行基线安装
cd ansible
ansible-playbook playbooks/setup_gpu_baseline.yml

# 4. 快速验证
ansible-playbook playbooks/validate_gpu.yml -e "level=quick"

# 5. 完整验证（交付前）
ansible-playbook playbooks/validate_gpu.yml -e "level=full"
```

### 日常维护

```bash
# 健康检查
./scripts/validation/quick_check.sh

# 持续监控
./scripts/monitoring/gpu_monitor.sh

# 定期验证
ansible-playbook playbooks/validate_gpu.yml -e "level=standard"
```

## 关键依赖和版本

### 系统要求
- OS: Ubuntu 20.04/22.04 或 RHEL 8/9
- Kernel: 5.x+
- Python: 3.8+

### 软件依赖
```yaml
- Ansible: >= 2.10
- nvidia-driver: >= 525.x
- CUDA: >= 11.8
- Docker: >= 20.10 (可选)
- DCGM: >= 3.0
```

### Python 包
```txt
pynvml
py3nvml
gputil
pandas
jinja2
```

## 下一步工作

1. **实现 Ansible Roles**
   - 创建 gpu_baseline role
   - 创建 gpu_validation role

2. **开发验证脚本**
   - quick_check.sh
   - standard_check.sh
   - full_validation.sh
   - gpu_health.py

3. **测试和优化**
   - 在测试环境验证
   - 性能优化
   - 错误处理完善

4. **文档完善**
   - 用户指南
   - 故障排除指南
   - API 文档

# GPU 集群部署和管理最佳实践

本文档提供基于本项目的 GPU 集群部署、配置、优化和维护的完整最佳实践指南。

## 目录

1. [概述](#概述)
2. [场景化部署指南](#场景化部署指南)
3. [初始部署流程](#初始部署流程)
4. [驱动安装最佳实践](#驱动安装最佳实践)
5. [性能优化最佳实践](#性能优化最佳实践)
6. [验证和测试最佳实践](#验证和测试最佳实践)
7. [监控和维护](#监控和维护)
8. [安全最佳实践](#安全最佳实践)
9. [故障排除流程](#故障排除流程)
10. [版本管理和升级](#版本管理和升级)
11. [文档和记录](#文档和记录)
12. [团队协作](#团队协作)

---

## 概述

### 项目功能总览

```
┌─────────────────────────────────────────────────────────────┐
│                    GPU 集群管理平台                          │
├─────────────────────────────────────────────────────────────┤
│  1. 驱动管理     │ Native/Container/Precompiled 三种方法   │
│  2. 自动检测     │ GPU 型号 → CUDA 版本自动匹配            │
│  3. CPU 优化     │ NUMA/Governor/Turbo/C-States           │
│  4. 系统验证     │ 8 大类完整验证                          │
│  5. 带宽测试     │ PCIe/NVLink/RDMA 测试                  │
│  6. 性能基准     │ NCCL/Megatron 训练基准                 │
│  7. NGC 管理     │ PyTorch/NeMo/Triton 镜像管理           │
│  8. 预编译驱动   │ 快速部署，93% 时间节省                  │
└─────────────────────────────────────────────────────────────┘
```

### 核心价值主张

- ⚡ **快速部署**: 预编译驱动实现 93% 时间节省
- 🎯 **智能配置**: 自动检测 GPU 并选择最佳 CUDA 版本
- 🚀 **性能提升**: CPU 优化带来 20-40% 性能提升
- ✅ **完整验证**: 8 大类验证确保系统就绪
- 📊 **基准测试**: 内置性能基线对比
- 🔧 **易于管理**: 统一的工具和接口

---

## 场景化部署指南

### 场景 1: 小型研发团队（1-10 台 GPU 服务器）

**特点**:
- 服务器数量少
- 管理相对简单
- 需要快速上手

**推荐配置**:

```yaml
# 部署策略
driver_installation_method: native
auto_detect_cuda_version: true
enable_cpu_optimization: true
enable_monitoring: basic

# NGC 镜像
ngc_images_to_pull:
  - pytorch:24.01
  - jupyter

# 验证级别
validation_level: quick
```

**部署命令**:

```bash
# 1. 克隆项目
git clone <repo-url>
cd gpu_passthrough

# 2. 配置 Ansible inventory
cat > ansible/inventory/hosts << EOF
[gpu_nodes]
gpu-server-01 ansible_host=192.168.1.101
gpu-server-02 ansible_host=192.168.1.102
EOF

# 3. 一键部署
cd ansible
ansible-playbook -i inventory/hosts playbooks/setup_gpu_baseline.yml

# 4. 验证
ansible -i inventory/hosts gpu_nodes -m shell -a "nvidia-smi"
```

**预期时间**: 30-45 分钟（包含重启）

---

### 场景 2: 中型 AI 训练集群（10-100 台服务器）

**特点**:
- 中等规模
- 需要自动化
- 性能要求高
- 需要监控

**推荐配置**:

```yaml
# 部署策略
driver_installation_method: precompiled  # 使用预编译驱动
auto_detect_cuda_version: true
enable_cpu_optimization: true
enable_monitoring: full

# CPU 优化
cpu_governor: performance
enable_turbo_boost: true
optimize_numa: true

# NGC 镜像
ngc_images_to_pull:
  - pytorch:24.01
  - nemo:24.01
  - tensorboard

# 验证和测试
validation_level: full
run_bandwidth_tests: true
run_nccl_tests: true

# 监控
enable_dcgm: true
enable_prometheus_exporter: true
```

**部署流程**:

```bash
# 阶段 1: 预构建预编译驱动（一次性）
./scripts/install/batch_build_drivers.sh

# 阶段 2: 部署到测试节点
ansible-playbook -i inventory/hosts playbooks/full_deployment_optimized.yml \
  --limit test_nodes

# 阶段 3: 验证测试节点
ansible-playbook -i inventory/hosts playbooks/validate_gpu.yml \
  --limit test_nodes

# 阶段 4: 运行基准测试
ssh test-node-01 "sudo /usr/local/bin/gpu-benchmark bandwidth"
ssh test-node-01 "sudo /usr/local/bin/gpu-benchmark nccl"

# 阶段 5: 批量部署到生产（分批）
for batch in batch1 batch2 batch3; do
  ansible-playbook -i inventory/hosts playbooks/full_deployment_optimized.yml \
    --limit $batch \
    --forks 10
  sleep 300  # 等待 5 分钟观察
done

# 阶段 6: 全集群验证
ansible-playbook -i inventory/hosts playbooks/validate_gpu.yml
```

**预期时间**:
- 预构建: 1-2 小时
- 部署（并行 10 节点）: 20-30 分钟/批次

---

### 场景 3: 大规模生产集群（100+ 台服务器）

**特点**:
- 大规模部署
- 高可用要求
- 严格的变更管理
- 需要完整的监控和告警

**推荐配置**:

```yaml
# 部署策略
driver_installation_method: precompiled
auto_detect_cuda_version: true
enable_cpu_optimization: true
enable_monitoring: enterprise

# 高可用配置
driver_rollback_enabled: true
health_check_interval: 300
auto_recovery: true

# 分阶段部署
deployment_strategy: canary  # 金丝雀部署
canary_percentage: 5
canary_validation_time: 3600  # 1 小时

# 监控告警
enable_dcgm: true
enable_prometheus: true
enable_grafana: true
alert_on_errors: true
alert_channels: ["slack", "email", "pagerduty"]

# 合规性
audit_logging: true
change_tracking: true
```

**企业级部署流程**:

```bash
# 1. 准备阶段
# 1.1 构建预编译驱动矩阵
./scripts/install/batch_build_drivers.sh

# 1.2 建立驱动仓库
rsync -av /opt/precompiled-drivers/ repo-server:/var/www/drivers/

# 1.3 验证仓库可访问性
curl http://repo-server/drivers/index.json

# 2. 金丝雀部署（5%）
ansible-playbook -i inventory/hosts playbooks/full_deployment_optimized.yml \
  --limit canary_nodes \
  --extra-vars "deployment_phase=canary"

# 3. 金丝雀验证（1小时）
for i in {1..12}; do
  ansible -i inventory/hosts canary_nodes -m shell \
    -a "/usr/local/bin/check-driver-health.sh"
  sleep 300
done

# 4. 蓝绿部署到生产
# 部署到蓝组（50%）
ansible-playbook -i inventory/hosts playbooks/full_deployment_optimized.yml \
  --limit blue_nodes \
  --forks 20

# 验证蓝组
./scripts/validation/cluster_health_check.sh blue_nodes

# 部署到绿组（剩余 50%）
ansible-playbook -i inventory/hosts playbooks/full_deployment_optimized.yml \
  --limit green_nodes \
  --forks 20

# 5. 全集群验证
ansible-playbook -i inventory/hosts playbooks/validate_gpu.yml \
  --extra-vars "validation_level=full"

# 6. 性能基准测试
ansible -i inventory/hosts all_gpu_nodes -m shell \
  -a "gpu-benchmark bandwidth" > benchmark_results.txt

# 7. 生成部署报告
./scripts/utils/generate_deployment_report.sh
```

**预期时间**:
- 准备: 2-4 小时
- 金丝雀: 2 小时（包含验证）
- 蓝绿部署: 3-5 小时
- 总计: ~8-12 小时

---

### 场景 4: Kubernetes GPU 集群

**特点**:
- 容器化环境
- 动态调度
- 使用 GPU Operator

**推荐配置**:

```yaml
# 驱动策略
driver_installation_method: driver-container
driver_container_image: nvcr.io/nvidia/driver
driver_container_tag: 535.154.05-ubuntu22.04

# GPU Operator 集成
use_gpu_operator: true
gpu_operator_version: v24.3.0

# NGC 镜像
ngc_images_to_pull:
  - pytorch:24.01
  - nemo:24.01
  - triton:24.01
```

**部署流程**:

```bash
# 1. 准备 Kubernetes 节点
ansible-playbook -i inventory/hosts playbooks/setup_gpu_baseline.yml \
  -e "driver_installation_method=driver-container"

# 2. 安装 GPU Operator
helm repo add nvidia https://nvidia.github.io/gpu-operator
helm repo update

helm install gpu-operator nvidia/gpu-operator \
  -n gpu-operator-resources \
  --create-namespace \
  -f gpu-operator-values.yaml

# 3. 验证 GPU Operator
kubectl get pods -n gpu-operator-resources

# 4. 测试 GPU 访问
kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: gpu-test
spec:
  containers:
  - name: cuda
    image: nvcr.io/nvidia/cuda:12.2.0-base-ubuntu22.04
    command: ["nvidia-smi"]
    resources:
      limits:
        nvidia.com/gpu: 1
EOF

kubectl logs gpu-test
```

---

## 初始部署流程

### 标准部署流程图

```
┌─────────────────┐
│  1. 环境准备     │
│  - 检查硬件      │
│  - 网络配置      │
│  - SSH 访问      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  2. 检测 GPU     │
│  - 自动检测型号  │
│  - 选择 CUDA 版本│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  3. 安装驱动     │
│  - Native/      │
│    Container/   │
│    Precompiled  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  4. CPU 优化     │
│  - NUMA 配置     │
│  - Governor 设置 │
│  - Turbo 启用    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  5. 系统验证     │
│  - 8 类检查      │
│  - 基准测试      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  6. NGC 镜像     │
│  - 拉取镜像      │
│  - 测试镜像      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  7. 交付验收     │
│  - 生成报告      │
│  - 移交文档      │
└─────────────────┘
```

### 详细部署步骤

#### 步骤 1: 环境准备清单

```bash
#!/bin/bash
# 环境准备检查脚本

echo "=== 环境准备检查 ==="

# 1. 检查硬件
check_hardware() {
    echo "1. 硬件检查"

    # CPU
    echo "  CPU: $(lscpu | grep "Model name" | cut -d: -f2 | xargs)"
    echo "  CPU Cores: $(nproc)"

    # 内存
    echo "  Memory: $(free -h | grep Mem | awk '{print $2}')"

    # 磁盘
    echo "  Disk: $(df -h / | tail -1 | awk '{print $2}')"

    # GPU
    if lspci | grep -qi nvidia; then
        echo "  GPU: $(lspci | grep -i nvidia | head -1 | cut -d: -f3)"
    else
        echo "  ✗ No NVIDIA GPU detected"
        return 1
    fi
}

# 2. 检查操作系统
check_os() {
    echo "2. 操作系统检查"

    . /etc/os-release
    echo "  OS: $PRETTY_NAME"
    echo "  Kernel: $(uname -r)"

    # 检查支持的操作系统
    if [[ "$ID" != "ubuntu" && "$ID" != "centos" && "$ID" != "rhel" ]]; then
        echo "  ⚠ Unsupported OS: $ID"
    fi
}

# 3. 检查网络
check_network() {
    echo "3. 网络检查"

    # Internet 访问
    if ping -c 1 8.8.8.8 &>/dev/null; then
        echo "  ✓ Internet access"
    else
        echo "  ✗ No internet access"
        return 1
    fi

    # NVIDIA 仓库访问
    if curl -s https://developer.download.nvidia.com &>/dev/null; then
        echo "  ✓ NVIDIA repository accessible"
    else
        echo "  ✗ NVIDIA repository not accessible"
    fi
}

# 4. 检查先决条件
check_prerequisites() {
    echo "4. 先决条件检查"

    local required_packages=("build-essential" "python3" "ansible")

    for pkg in "${required_packages[@]}"; do
        if dpkg -l | grep -q "^ii  $pkg"; then
            echo "  ✓ $pkg installed"
        else
            echo "  ✗ $pkg not installed"
        fi
    done
}

# 执行所有检查
check_hardware
check_os
check_network
check_prerequisites

echo ""
echo "=== 检查完成 ==="
```

#### 步骤 2: 配置 Ansible Inventory

```ini
# ansible/inventory/hosts

[all:vars]
ansible_user=ubuntu
ansible_become=true
ansible_python_interpreter=/usr/bin/python3

# GPU 节点组
[gpu_nodes]
gpu-01 ansible_host=192.168.1.101 gpu_type=A100
gpu-02 ansible_host=192.168.1.102 gpu_type=A100
gpu-03 ansible_host=192.168.1.103 gpu_type=H100

# 环境分组
[dev]
gpu-01

[staging]
gpu-02

[production]
gpu-03

# 角色分组
[training_nodes]
gpu-[01:02]

[inference_nodes]
gpu-03
```

#### 步骤 3: 执行部署

```bash
# 基础部署（推荐新手）
ansible-playbook -i inventory/hosts playbooks/setup_gpu_baseline.yml

# 完整优化部署（推荐生产）
ansible-playbook -i inventory/hosts playbooks/full_deployment_optimized.yml

# 自定义部署
ansible-playbook -i inventory/hosts playbooks/full_deployment_optimized.yml \
  -e "driver_installation_method=precompiled" \
  -e "enable_cpu_optimization=true" \
  -e "validation_level=full" \
  --limit production
```

---

## 驱动安装最佳实践

### 方法选择决策树

```
开始
  │
  ├─ 是否使用 Kubernetes?
  │  ├─ 是 → 使用 Driver Container
  │  └─ 否 → 继续
  │
  ├─ 集群规模?
  │  ├─ > 50 节点 → 使用 Precompiled
  │  ├─ 10-50 节点 → 考虑 Precompiled
  │  └─ < 10 节点 → Native 或 Precompiled
  │
  ├─ 内核版本统一?
  │  ├─ 是 → Precompiled（强烈推荐）
  │  └─ 否 → Native 或为每个版本构建
  │
  ├─ 是否需要快速回滚?
  │  ├─ 是 → Driver Container 或 Precompiled
  │  └─ 否 → Native
  │
  └─ 最终决策
```

### Native 安装最佳实践

**适用场景**:
- 小规模部署（< 10 节点）
- 传统数据中心
- 不需要频繁更新

**配置建议**:

```yaml
# ansible/roles/gpu_baseline/defaults/main.yml
driver_installation_method: native
auto_detect_cuda_version: true
nvidia_driver_version: "535"  # 会被自动检测覆盖
cuda_version: "12-2"

# 重启配置
nvidia_driver_skip_reboot: false
reboot_timeout: 600

# 验证
run_post_install_validation: true
```

**注意事项**:

1. **编译时间**: 预留 20-30 分钟/节点
2. **内核更新**: 内核更新后需重新编译
3. **依赖管理**: 确保 build-essential 和 kernel-headers 已安装

```bash
# 预安装依赖
ansible -i inventory/hosts gpu_nodes -m apt -a \
  "name=build-essential,linux-headers-$(uname -r) state=present"

# 安装驱动
ansible-playbook -i inventory/hosts playbooks/setup_gpu_baseline.yml

# 验证
ansible -i inventory/hosts gpu_nodes -m shell -a "nvidia-smi"
```

### Precompiled 安装最佳实践

**适用场景**:
- 大规模部署（> 50 节点）
- 需要快速部署
- 内核版本统一或可控

**完整流程**:

```bash
# 1. 识别所有内核版本
ansible -i inventory/hosts all -m shell \
  -a "uname -r" | grep -v ">>" | sort -u > kernels.txt

# 2. 批量构建预编译驱动
cat > build_config.sh << 'EOF'
#!/bin/bash
DRIVER_VERSIONS=("535.154.05" "550.90.07")
KERNEL_VERSIONS=($(cat kernels.txt))

export OUTPUT_DIR=/opt/precompiled-drivers
export USE_CONTAINER=true

./scripts/install/batch_build_drivers.sh
EOF

bash build_config.sh

# 3. 建立驱动仓库
mkdir -p /var/www/drivers
cp -r /opt/precompiled-drivers/* /var/www/drivers/
./scripts/utils/manage_precompiled_drivers.sh update-index

# 4. 启动 HTTP 服务
cd /var/www/drivers
python3 -m http.server 8080 &

# 5. 批量部署
ansible-playbook -i inventory/hosts playbooks/setup_gpu_baseline.yml \
  -e "driver_installation_method=precompiled" \
  -e "precompiled_repo=http://repo-server:8080" \
  --forks 20

# 6. 验证
ansible -i inventory/hosts all -m shell \
  -a "nvidia-smi && /usr/local/bin/gpu-benchmark verify"
```

**关键配置**:

```yaml
# 预编译驱动配置
driver_installation_method: precompiled
precompiled_repo: "http://internal-repo.company.com/drivers"
use_precompiled: true

# 自动选择匹配的驱动
auto_match_kernel: true

# 回滚配置
enable_driver_backup: true
backup_retention_days: 30
```

### Driver Container 最佳实践

**适用场景**:
- Kubernetes 集群
- 需要版本隔离
- 需要快速切换版本

**配置**:

```yaml
driver_installation_method: driver-container
driver_container_image: nvcr.io/nvidia/driver
driver_container_tag: 535.154.05-ubuntu22.04
driver_container_enable_persistence: true

# Health check 配置
driver_health_check_interval: 300
driver_auto_restart: true
```

**管理命令**:

```bash
# 查看驱动容器状态
systemctl status nvidia-driver

# 查看日志
journalctl -u nvidia-driver -f

# 重启驱动容器
systemctl restart nvidia-driver

# 切换驱动版本
# 1. 停止当前容器
systemctl stop nvidia-driver

# 2. 更新配置
sudo sed -i 's/535.154.05/550.90.07/g' /etc/systemd/system/nvidia-driver.service

# 3. 重新加载并启动
systemctl daemon-reload
systemctl start nvidia-driver

# 4. 验证
nvidia-smi
```

---

## 性能优化最佳实践

### CPU 优化配置

**推荐配置**（生产环境）:

```yaml
# ansible/roles/cpu_optimization/defaults/main.yml

# CPU Governor（最重要）
cpu_governor: performance  # 强制性能模式

# Turbo Boost
enable_turbo_boost: true  # Intel Turbo Boost / AMD Precision Boost

# NUMA 优化
optimize_numa: true
numa_balancing: false  # 禁用自动 NUMA 平衡

# C-States（降低延迟）
c_states_config:
  max_cstate: 1  # 限制到 C1
  disable_deep_sleep: true

# IOMMU
kernel_params:
  intel_iommu: "on"
  iommu: "pt"  # passthrough 模式
  pcie_aspm: "off"  # 禁用 PCIe 电源管理

# 内存优化
vm_swappiness: 10  # 减少 swap 使用
transparent_hugepages: madvise

# IRQ 亲和性
configure_irq_affinity: true
```

**验证优化效果**:

```bash
# 运行系统检查
./scripts/validation/system_check.sh

# 预期输出示例：
# ✓ CPU Governor: performance (on all cores)
# ✓ Turbo Boost: enabled
# ✓ NUMA nodes: 2
# ✓ GPU 0 on NUMA node 0
# ✓ C-States: C1 only
# ✓ IOMMU: enabled (passthrough mode)

# 性能对比测试
# 优化前
./scripts/benchmarks/nccl_benchmark.sh > before.txt

# 应用优化
ansible-playbook -i inventory/hosts playbooks/apply_cpu_optimization.yml

# 优化后
./scripts/benchmarks/nccl_benchmark.sh > after.txt

# 对比结果
diff before.txt after.txt
```

**预期性能提升**:

| 工作负载类型 | 优化前 | 优化后 | 提升 |
|------------|--------|--------|------|
| NCCL AllReduce | 180 GB/s | 245 GB/s | 36% |
| 训练吞吐量 | 2500 samples/s | 3200 samples/s | 28% |
| GPU 利用率 | 75% | 92% | 23% |

### NUMA 配置最佳实践

**检查 NUMA 拓扑**:

```bash
# 查看 NUMA 节点
numactl --hardware

# 查看 GPU-NUMA 映射
nvidia-smi topo -m

# 检查 GPU 亲和性
for gpu in $(seq 0 7); do
  echo "GPU $gpu: NUMA node $(cat /sys/class/drm/card$gpu/device/numa_node)"
done
```

**正确的 NUMA 绑定**:

```bash
#!/bin/bash
# 运行训练时绑定到正确的 NUMA 节点

GPU_ID=0
NUMA_NODE=$(cat /sys/class/drm/card${GPU_ID}/device/numa_node)

# 启动训练，绑定到对应的 NUMA 节点
numactl --cpunodebind=${NUMA_NODE} --membind=${NUMA_NODE} \
  python train.py --gpu ${GPU_ID}
```

### 网络优化（多节点训练）

**InfiniBand 优化**:

```bash
# /etc/modprobe.d/mlx5_core.conf
options mlx5_core log_max_qp=20

# 优化 IB 参数
echo 1 > /sys/class/net/ib0/mode
echo 65520 > /sys/class/net/ib0/mtu

# 验证
ibstat
ibstatus
```

**RoCE 优化**:

```bash
# 启用 PFC (Priority Flow Control)
mlnx_qos -i ens1f0 --pfc 0,0,0,1,0,0,0,0

# 设置 ECN
mlnx_qos -i ens1f0 --trust dscp

# 配置 DSCP
echo 4 > /sys/class/infiniband/mlx5_0/tc/1/traffic_class
```

---

## 验证和测试最佳实践

### 三级验证策略

#### Level 1: 快速验证（5 分钟）

**目的**: 确认基本功能正常

```bash
# 运行快速检查
./scripts/validation/quick_check.sh

# 或通过 Ansible
ansible -i inventory/hosts gpu_nodes -m script \
  -a "./scripts/validation/quick_check.sh"
```

**检查项**:
- ✓ nvidia-smi 可用
- ✓ GPU 检测
- ✓ 驱动版本
- ✓ 温度正常
- ✓ 无 ECC 错误

#### Level 2: 标准验证（15 分钟）

**目的**: 全面系统检查

```bash
# 运行系统检查
./scripts/validation/system_check.sh

# 生成 JSON 报告
./scripts/validation/system_check.sh --json > system_report.json
```

**检查项**（8 大类）:
1. CPU 配置
2. NUMA 配置
3. IOMMU 配置
4. PCIe 配置
5. GPU 配置
6. 内存配置
7. 内核参数
8. 容器运行时

#### Level 3: 完整验证（30-60 分钟）

**目的**: 性能基准测试

```bash
# 1. 带宽测试
./scripts/validation/bandwidth_test.sh

# 2. NCCL 测试
./scripts/benchmarks/nccl_benchmark.sh

# 3. 训练基准（可选）
MODEL_SIZE=GPT-1.2B ./scripts/benchmarks/megatron_benchmark.sh

# 4. 压力测试
./scripts/validation/stress_test.sh --duration 3600  # 1 小时
```

### 自动化验证流程

```yaml
# playbooks/comprehensive_validation.yml

- name: Comprehensive GPU Cluster Validation
  hosts: gpu_nodes
  become: yes
  serial: "20%"  # 每次 20% 节点

  tasks:
    - name: Level 1 - Quick Check
      script: ../scripts/validation/quick_check.sh
      register: quick_check

    - name: Level 2 - System Check
      script: ../scripts/validation/system_check.sh --json
      register: system_check

    - name: Save reports
      copy:
        content: "{{ system_check.stdout }}"
        dest: "/var/log/validation_{{ inventory_hostname }}_{{ ansible_date_time.epoch }}.json"

    - name: Level 3 - Bandwidth Test
      script: ../scripts/validation/bandwidth_test.sh
      when: validation_level == "full"

    - name: Level 3 - NCCL Benchmark
      script: ../scripts/benchmarks/nccl_benchmark.sh
      when: validation_level == "full"
```

**运行验证**:

```bash
# 快速验证
ansible-playbook -i inventory/hosts playbooks/comprehensive_validation.yml \
  -e "validation_level=quick"

# 完整验证
ansible-playbook -i inventory/hosts playbooks/comprehensive_validation.yml \
  -e "validation_level=full"

# 收集报告
ansible -i inventory/hosts all -m fetch \
  -a "src=/var/log/validation_*.json dest=./reports/"
```

### 基准测试基线建立

**建立集群基线**:

```bash
#!/bin/bash
# establish_baseline.sh

CLUSTER_NAME="production"
BASELINE_DIR="/opt/baselines/${CLUSTER_NAME}"

mkdir -p "${BASELINE_DIR}"

# 1. 收集系统信息
for node in $(ansible -i inventory/hosts gpu_nodes --list-hosts | grep -v hosts); do
  echo "Collecting baseline from $node..."

  # 系统配置
  ssh $node "nvidia-smi -q" > "${BASELINE_DIR}/${node}_gpu_info.txt"

  # 带宽测试
  ssh $node "./scripts/validation/bandwidth_test.sh" > "${BASELINE_DIR}/${node}_bandwidth.json"

  # NCCL 测试
  ssh $node "./scripts/benchmarks/nccl_benchmark.sh" > "${BASELINE_DIR}/${node}_nccl.json"
done

# 2. 生成基线报告
python3 << EOF
import json
import glob

baselines = {}
for file in glob.glob("${BASELINE_DIR}/*_bandwidth.json"):
    node = file.split('/')[-1].split('_')[0]
    with open(file) as f:
        baselines[node] = json.load(f)

with open("${BASELINE_DIR}/cluster_baseline.json", 'w') as f:
    json.dump(baselines, f, indent=2)

print(f"Baseline established for {len(baselines)} nodes")
EOF

echo "Baseline saved to: ${BASELINE_DIR}"
```

**使用基线对比**:

```bash
# 运行当前测试
./scripts/validation/bandwidth_test.sh > current_test.json

# 对比基线
python3 << EOF
import json

with open('current_test.json') as f:
    current = json.load(f)

with open('/opt/baselines/production/gpu-01_bandwidth.json') as f:
    baseline = json.load(f)

# 对比关键指标
for metric in ['pcie_bandwidth', 'nvlink_bandwidth']:
    current_val = current.get(metric)
    baseline_val = baseline.get(metric)

    if current_val and baseline_val:
        diff = (current_val - baseline_val) / baseline_val * 100
        status = "✓" if abs(diff) < 5 else "⚠"
        print(f"{status} {metric}: {current_val:.2f} GB/s (baseline: {baseline_val:.2f}, diff: {diff:+.1f}%)")
EOF
```

---

## 监控和维护

### 监控架构

```
┌─────────────────────────────────────────────────┐
│              Monitoring Stack                   │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │  DCGM    │→ │Prometheus│→ │ Grafana  │     │
│  │ Exporter │  │          │  │          │     │
│  └──────────┘  └──────────┘  └──────────┘     │
│       ↑                            ↓           │
│  ┌──────────────┐          ┌─────────────┐    │
│  │ GPU Metrics  │          │ Alertmanager│    │
│  │ - Utilization│          │ - Slack     │    │
│  │ - Temperature│          │ - Email     │    │
│  │ - Memory     │          │ - PagerDuty │    │
│  │ - Power      │          └─────────────┘    │
│  │ - ECC Errors │                              │
│  └──────────────┘                              │
└─────────────────────────────────────────────────┘
```

### 关键监控指标

**GPU 指标**:

```yaml
# 必须监控的指标
gpu_metrics:
  - gpu_utilization        # > 80% (训练时)
  - memory_utilization     # < 95%
  - temperature           # < 85°C
  - power_usage           # 符合预期
  - ecc_errors            # = 0
  - pcie_throughput       # 符合基线
  - sm_clock              # 符合预期
  - memory_clock          # 符合预期

# 告警阈值
alerts:
  high_temperature:
    threshold: 85
    severity: warning
  critical_temperature:
    threshold: 90
    severity: critical
  ecc_errors:
    threshold: 1
    severity: critical
  low_utilization:
    threshold: 20
    duration: 1h
    severity: info
```

### DCGM 监控部署

```bash
# 1. 安装 DCGM
ansible -i inventory/hosts gpu_nodes -m apt \
  -a "name=datacenter-gpu-manager state=present"

# 2. 启动 DCGM
ansible -i inventory/hosts gpu_nodes -m service \
  -a "name=nvidia-dcgm state=started enabled=yes"

# 3. 部署 DCGM Exporter
cat > dcgm-exporter.yaml << EOF
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: dcgm-exporter
  namespace: gpu-monitoring
spec:
  selector:
    matchLabels:
      app: dcgm-exporter
  template:
    metadata:
      labels:
        app: dcgm-exporter
    spec:
      containers:
      - name: dcgm-exporter
        image: nvcr.io/nvidia/k8s/dcgm-exporter:3.1.7-3.1.4-ubuntu22.04
        ports:
        - containerPort: 9400
          name: metrics
        securityContext:
          privileged: true
        volumeMounts:
        - name: pod-resources
          mountPath: /var/lib/kubelet/pod-resources
      volumes:
      - name: pod-resources
        hostPath:
          path: /var/lib/kubelet/pod-resources
EOF

kubectl apply -f dcgm-exporter.yaml

# 4. 配置 Prometheus 抓取
cat >> prometheus.yml << EOF
scrape_configs:
  - job_name: 'dcgm'
    kubernetes_sd_configs:
    - role: pod
    relabel_configs:
    - source_labels: [__meta_kubernetes_pod_label_app]
      action: keep
      regex: dcgm-exporter
EOF
```

### Grafana 仪表板

**推荐仪表板**:

1. **NVIDIA DCGM Exporter Dashboard** (ID: 12239)
2. **GPU Cluster Dashboard** (自定义)

```json
{
  "dashboard": {
    "title": "GPU Cluster Overview",
    "panels": [
      {
        "title": "GPU Utilization",
        "targets": [{
          "expr": "DCGM_FI_DEV_GPU_UTIL"
        }]
      },
      {
        "title": "GPU Temperature",
        "targets": [{
          "expr": "DCGM_FI_DEV_GPU_TEMP"
        }]
      },
      {
        "title": "GPU Memory Usage",
        "targets": [{
          "expr": "DCGM_FI_DEV_FB_USED / DCGM_FI_DEV_FB_FREE * 100"
        }]
      },
      {
        "title": "ECC Errors",
        "targets": [{
          "expr": "DCGM_FI_DEV_ECC_DBE_VOL_TOTAL"
        }]
      }
    ]
  }
}
```

### 日常维护检查清单

**每日检查**（自动化）:

```bash
#!/bin/bash
# daily_health_check.sh

# 1. GPU 健康检查
nvidia-smi --query-gpu=index,name,driver_version,temperature.gpu,utilization.gpu,memory.used,memory.total,ecc.errors.uncorrected.aggregate.total \
  --format=csv,noheader

# 2. 驱动状态
if ! nvidia-smi &>/dev/null; then
  echo "ALERT: nvidia-smi failed"
  exit 1
fi

# 3. 温度检查
MAX_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | sort -n | tail -1)
if [ "$MAX_TEMP" -gt 85 ]; then
  echo "ALERT: High temperature detected: ${MAX_TEMP}°C"
fi

# 4. ECC 错误检查
ECC_ERRORS=$(nvidia-smi --query-gpu=ecc.errors.uncorrected.aggregate.total --format=csv,noheader | awk '{s+=$1} END {print s}')
if [ "$ECC_ERRORS" -gt 0 ]; then
  echo "ALERT: ECC errors detected: $ECC_ERRORS"
fi

# 5. PCIe 检查
./scripts/validation/bandwidth_test.sh --quick

# 6. 磁盘空间
DISK_USAGE=$(df / | tail -1 | awk '{print $5}' | tr -d '%')
if [ "$DISK_USAGE" -gt 80 ]; then
  echo "WARNING: Disk usage high: ${DISK_USAGE}%"
fi
```

**每周检查**（手动或自动）:

```bash
#!/bin/bash
# weekly_maintenance.sh

# 1. 完整系统验证
./scripts/validation/system_check.sh --json > weekly_report_$(date +%Y%m%d).json

# 2. 性能基线对比
./scripts/benchmarks/nccl_benchmark.sh > nccl_weekly.txt
diff nccl_baseline.txt nccl_weekly.txt

# 3. 驱动日志检查
dmesg | grep -i nvidia | grep -i error

# 4. 包更新检查
apt list --upgradable | grep nvidia

# 5. 清理临时文件
find /tmp -name "nvidia*" -mtime +7 -delete
find /var/log -name "nvidia*" -mtime +30 -delete
```

**每月检查**:

- 完整基准测试
- 驱动版本评估
- 安全更新
- 容量规划
- 文档更新

---

## 安全最佳实践

### 访问控制

**用户权限管理**:

```bash
# 创建 GPU 用户组
groupadd gpuusers

# 添加用户到组
usermod -aG gpuusers alice
usermod -aG gpuusers bob

# 配置 nvidia-smi 权限
cat > /etc/udev/rules.d/99-nvidia.rules << EOF
KERNEL=="nvidia*", GROUP="gpuusers", MODE="0660"
KERNEL=="nvidiactl", GROUP="gpuusers", MODE="0660"
KERNEL=="nvidia-modeset", GROUP="gpuusers", MODE="0660"
KERNEL=="nvidia-uvm", GROUP="gpuusers", MODE="0660"
EOF

# 重新加载 udev
udevadm control --reload-rules
udevadm trigger
```

**容器隔离**:

```bash
# 限制容器可见的 GPU
docker run --gpus '"device=0,1"' ...  # 只能访问 GPU 0 和 1

# 使用 MIG (Multi-Instance GPU) 进行硬件隔离
nvidia-smi mig -cgi 19,19,19 -C  # 创建 3 个 MIG 实例
```

### 安全加固

**禁用不必要的功能**:

```bash
# /etc/modprobe.d/nvidia-security.conf

# 禁用 persistence mode（如果不需要）
# options nvidia NVreg_EnablePersistenced=0

# 启用安全模式
options nvidia NVreg_EnableGpuFirmware=1

# 禁用调试
options nvidia NVreg_EnableDbgBreakpoint=0
```

**审计日志**:

```bash
# 启用审计
auditctl -w /dev/nvidia0 -p wa -k gpu_access
auditctl -w /usr/bin/nvidia-smi -p x -k gpu_tools

# 查看审计日志
ausearch -k gpu_access
```

### 网络安全

**防火墙配置**:

```bash
# UFW 配置（Ubuntu）
# 只允许内部网络访问 NCCL 端口
ufw allow from 192.168.0.0/16 to any port 50000:51000 proto tcp

# 允许 Prometheus 抓取
ufw allow from monitoring-server to any port 9400 proto tcp
```

**SSH 加固**:

```bash
# /etc/ssh/sshd_config
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
AllowUsers gpu-admin@192.168.0.0/16
```

---

## 故障排除流程

### 故障分类和处理流程

```
故障报告
    │
    ├─ 驱动问题？
    │  ├─ nvidia-smi 失败 → 检查驱动加载 → 重启驱动/重新安装
    │  ├─ 版本不匹配 → 检查 CUDA 兼容性 → 更新驱动
    │  └─ 模块加载失败 → 检查内核版本 → 重新编译
    │
    ├─ 性能问题？
    │  ├─ GPU 利用率低 → 检查 CPU/NUMA → 优化配置
    │  ├─ 带宽低 → 运行 bandwidth_test → 检查 PCIe/NVLink
    │  └─ 训练慢 → 运行 NCCL 测试 → 检查网络
    │
    ├─ 硬件问题？
    │  ├─ 高温 → 检查散热 → 清理/RMA
    │  ├─ ECC 错误 → 检查日志 → 隔离/RMA
    │  └─ GPU 掉线 → 检查 PCIe → 重新插拔/RMA
    │
    └─ 其他问题？
       ├─ 容器访问失败 → 检查 runtime → 重新配置
       ├─ 多节点通信失败 → 检查网络 → 配置 RDMA
       └─ 权限问题 → 检查用户组 → 调整权限
```

### 常见问题快速解决

#### 问题 1: nvidia-smi 无响应

```bash
# 诊断
sudo dmesg | grep -i nvidia | tail -20
lsmod | grep nvidia
ls -la /dev/nvidia*

# 解决方案 1: 重新加载模块
sudo rmmod nvidia_drm nvidia_modeset nvidia_uvm nvidia
sudo modprobe nvidia

# 解决方案 2: 重启驱动服务（Driver Container）
sudo systemctl restart nvidia-driver

# 解决方案 3: 重新安装驱动
./scripts/utils/manage_precompiled_drivers.sh rollback
```

#### 问题 2: GPU 温度过高

```bash
# 检查温度
nvidia-smi --query-gpu=temperature.gpu,temperature.memory --format=csv

# 检查风扇
nvidia-smi --query-gpu=fan.speed --format=csv

# 临时降低功率限制
sudo nvidia-smi -pl 250  # 设置为 250W

# 检查数据中心环境
sensors  # 检查服务器温度
```

#### 问题 3: 性能下降

```bash
# 1. 运行完整诊断
./scripts/validation/system_check.sh > diagnostic.txt

# 2. 对比基线
./scripts/validation/bandwidth_test.sh > current_bandwidth.json
diff baseline_bandwidth.json current_bandwidth.json

# 3. 检查 CPU 配置
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
cat /sys/devices/system/cpu/intel_pstate/no_turbo

# 4. 检查 NUMA
numactl --hardware
nvidia-smi topo -m

# 5. 重新应用优化
ansible-playbook -i inventory/hosts playbooks/apply_cpu_optimization.yml
```

### 故障排除工具包

```bash
#!/bin/bash
# gpu_troubleshoot.sh - 一键故障诊断

echo "=== GPU Troubleshooting Toolkit ==="
echo ""

# 1. 基础检查
echo "1. Basic Checks"
echo "  Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
echo "  CUDA: $(nvcc --version 2>/dev/null | grep release | awk '{print $5}')"
echo "  GPUs: $(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)"

# 2. 模块检查
echo ""
echo "2. Kernel Modules"
lsmod | grep nvidia

# 3. 设备文件
echo ""
echo "3. Device Nodes"
ls -la /dev/nvidia* 2>/dev/null || echo "  No device nodes found"

# 4. PCIe 检查
echo ""
echo "4. PCIe Status"
nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv

# 5. 温度和功率
echo ""
echo "5. Temperature and Power"
nvidia-smi --query-gpu=temperature.gpu,power.draw,power.limit --format=csv

# 6. ECC 错误
echo ""
echo "6. ECC Errors"
nvidia-smi --query-gpu=ecc.errors.uncorrected.aggregate.total --format=csv

# 7. 进程列表
echo ""
echo "7. GPU Processes"
nvidia-smi pmon -c 1

# 8. 拓扑
echo ""
echo "8. GPU Topology"
nvidia-smi topo -m

# 9. 系统日志
echo ""
echo "9. Recent Errors (dmesg)"
dmesg | grep -i nvidia | grep -i error | tail -10

# 10. 建议
echo ""
echo "10. Recommendations"

# 检查驱动版本
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
if [[ "$DRIVER_VER" < "535" ]]; then
    echo "  ⚠ Consider upgrading driver to 535+ for better performance"
fi

# 检查温度
MAX_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | sort -n | tail -1)
if [ "$MAX_TEMP" -gt 80 ]; then
    echo "  ⚠ High temperature detected: ${MAX_TEMP}°C"
    echo "    - Check cooling system"
    echo "    - Consider reducing power limit"
fi

# 检查 ECC
ECC=$(nvidia-smi --query-gpu=ecc.errors.uncorrected.aggregate.total --format=csv,noheader | awk '{s+=$1} END {print s}')
if [ "$ECC" -gt 0 ]; then
    echo "  ✗ ECC errors detected: $ECC"
    echo "    - Run memory test"
    echo "    - Contact vendor if errors persist"
fi

echo ""
echo "=== Diagnostic Complete ==="
```

---

## 版本管理和升级

### 驱动升级策略

**金丝雀升级流程**:

```bash
#!/bin/bash
# canary_upgrade.sh

CANARY_NODES="gpu-test-01"
BLUE_NODES="gpu-prod-[01-10]"
GREEN_NODES="gpu-prod-[11-20]"

NEW_DRIVER="550.90.07"
VALIDATION_TIME=3600  # 1小时

# 1. 金丝雀部署
echo "Phase 1: Canary Deployment"
ansible-playbook -i inventory/hosts playbooks/upgrade_driver.yml \
  -e "driver_version=${NEW_DRIVER}" \
  --limit "${CANARY_NODES}"

# 2. 金丝雀验证
echo "Phase 2: Canary Validation (${VALIDATION_TIME}s)"
for i in $(seq 1 12); do
  ansible -i inventory/hosts canary -m script \
    -a "./scripts/validation/quick_check.sh"

  # 检查监控指标
  curl "http://prometheus:9090/api/v1/query?query=DCGM_FI_DEV_GPU_TEMP" | \
    jq '.data.result[] | select(.metric.instance == "gpu-test-01")'

  sleep 300
done

# 3. 用户确认
read -p "Canary successful? Continue to production? (yes/no) " -r
if [[ ! $REPLY =~ ^[Yy]es$ ]]; then
    echo "Upgrade cancelled"
    exit 1
fi

# 4. 蓝组升级
echo "Phase 3: Blue Group Upgrade"
ansible-playbook -i inventory/hosts playbooks/upgrade_driver.yml \
  -e "driver_version=${NEW_DRIVER}" \
  --limit "${BLUE_NODES}" \
  --serial 5  # 每次 5 台

# 5. 验证蓝组
ansible -i inventory/hosts blue -m script \
  -a "./scripts/validation/system_check.sh"

# 6. 绿组升级
echo "Phase 4: Green Group Upgrade"
ansible-playbook -i inventory/hosts playbooks/upgrade_driver.yml \
  -e "driver_version=${NEW_DRIVER}" \
  --limit "${GREEN_NODES}" \
  --serial 5

# 7. 全集群验证
echo "Phase 5: Full Cluster Validation"
ansible-playbook -i inventory/hosts playbooks/comprehensive_validation.yml

echo "Upgrade Complete!"
```

### 回滚流程

```bash
# 方法 1: 使用管理脚本（预编译驱动）
./scripts/utils/manage_precompiled_drivers.sh rollback

# 方法 2: 使用 Ansible
ansible-playbook -i inventory/hosts playbooks/rollback_driver.yml

# 方法 3: 手动回滚（Driver Container）
systemctl stop nvidia-driver
# 修改 /etc/systemd/system/nvidia-driver.service 中的版本
systemctl daemon-reload
systemctl start nvidia-driver
```

### 变更管理清单

**升级前**:
- [ ] 备份当前配置
- [ ] 记录当前驱动版本
- [ ] 运行基线测试
- [ ] 通知用户计划维护窗口
- [ ] 准备回滚方案

**升级中**:
- [ ] 遵循金丝雀流程
- [ ] 实时监控关键指标
- [ ] 记录所有操作
- [ ] 保持通信畅通

**升级后**:
- [ ] 验证所有节点
- [ ] 运行性能测试
- [ ] 对比基线
- [ ] 更新文档
- [ ] 通知用户完成

---

## 文档和记录

### 文档结构

```
/opt/gpu-cluster-docs/
├── inventory/
│   ├── hardware_inventory.xlsx      # 硬件清单
│   ├── software_versions.md         # 软件版本
│   └── network_topology.pdf         # 网络拓扑
├── configurations/
│   ├── driver_config.yaml           # 驱动配置
│   ├── optimization_params.yaml     # 优化参数
│   └── monitoring_config.yaml       # 监控配置
├── baselines/
│   ├── performance_baseline.json    # 性能基线
│   ├── bandwidth_baseline.json      # 带宽基线
│   └── training_baseline.json       # 训练基线
├── procedures/
│   ├── deployment_sop.md            # 部署 SOP
│   ├── upgrade_procedure.md         # 升级流程
│   ├── troubleshooting_guide.md     # 故障排除
│   └── emergency_response.md        # 应急响应
└── reports/
    ├── weekly_health_reports/       # 周报
    ├── monthly_performance/         # 月度性能
    └── incident_reports/            # 事件报告
```

### 自动化报告生成

```bash
#!/bin/bash
# generate_weekly_report.sh

REPORT_DATE=$(date +%Y%m%d)
REPORT_DIR="/opt/gpu-cluster-docs/reports/weekly_health_reports"
REPORT_FILE="${REPORT_DIR}/report_${REPORT_DATE}.md"

mkdir -p "${REPORT_DIR}"

cat > "${REPORT_FILE}" << EOF
# GPU Cluster Weekly Health Report
**Report Date**: $(date +%Y-%m-%d)
**Report Period**: $(date -d '7 days ago' +%Y-%m-%d) to $(date +%Y-%m-%d)

## Executive Summary

### Cluster Statistics
- Total Nodes: $(ansible -i inventory/hosts gpu_nodes --list-hosts | wc -l)
- Total GPUs: $(ansible -i inventory/hosts gpu_nodes -m shell -a "nvidia-smi --query-gpu=count --format=csv,noheader" | grep -v ">>" | awk '{s+=$1} END {print s}')
- Average GPU Utilization: TBD
- Average Temperature: TBD

### Alerts This Week
$(grep "$(date -d '7 days ago' +%Y-%m-%d)" /var/log/alerts.log | wc -l) alerts

## Node Status

EOF

# 收集每个节点状态
ansible -i inventory/hosts gpu_nodes -m script \
  -a "./scripts/validation/quick_check.sh" >> "${REPORT_FILE}"

# 性能指标
cat >> "${REPORT_FILE}" << EOF

## Performance Metrics

### Bandwidth Tests
$(cat /tmp/weekly_bandwidth_results.txt)

### NCCL Performance
$(cat /tmp/weekly_nccl_results.txt)

## Issues and Resolutions

### Critical Issues
- None

### Warnings
$(grep WARN /var/log/system_check.log | tail -10)

## Maintenance Activities

### Completed
- Weekly health check
- Performance baseline validation

### Planned
- Monthly driver update review (next week)

## Recommendations

1. All systems operating within normal parameters
2. No immediate action required

---
*Report generated automatically by gpu-cluster-tools*
EOF

echo "Report generated: ${REPORT_FILE}"

# 发送报告
mail -s "GPU Cluster Weekly Report - $(date +%Y-%m-%d)" \
  team@company.com < "${REPORT_FILE}"
```

---

## 团队协作

### 角色和职责

**GPU 集群管理员**:
- 部署和维护 GPU 基础设施
- 驱动和软件更新
- 性能优化
- 故障排除

**DevOps 工程师**:
- CI/CD 集成
- 自动化脚本
- 监控和告警
- 容器编排

**数据科学家/研究员**:
- 使用 GPU 资源
- 报告性能问题
- 提供优化反馈

**系统管理员**:
- 网络配置
- 存储管理
- 安全加固
- 备份恢复

### 沟通渠道

**Slack 集成**:

```bash
# 发送告警到 Slack
send_slack_alert() {
    local message=$1
    local severity=$2

    curl -X POST https://hooks.slack.com/services/YOUR/WEBHOOK/URL \
      -H 'Content-Type: application/json' \
      -d "{
        \"text\": \"GPU Alert\",
        \"attachments\": [{
          \"color\": \"${severity}\",
          \"text\": \"${message}\",
          \"ts\": $(date +%s)
        }]
      }"
}

# 使用示例
if [ "$GPU_TEMP" -gt 85 ]; then
    send_slack_alert "High GPU temperature: ${GPU_TEMP}°C on $(hostname)" "danger"
fi
```

### 知识库维护

**Wiki 结构**（Confluence/GitBook）:

```
GPU Cluster Wiki
├── Getting Started
│   ├── Quick Start Guide
│   ├── Access Request
│   └── First Job Tutorial
├── User Guides
│   ├── PyTorch on GPUs
│   ├── Multi-GPU Training
│   └── NGC Containers
├── Admin Guides
│   ├── Deployment Guide
│   ├── Upgrade Procedure
│   └── Troubleshooting
├── Reference
│   ├── Hardware Specs
│   ├── Software Versions
│   └── Performance Baselines
└── FAQ
    ├── Common Issues
    └── Best Practices
```

---

## 总结

### 关键要点

1. **选择正确的方法**
   - 小规模 → Native
   - 大规模 → Precompiled
   - Kubernetes → Driver Container

2. **自动化一切**
   - 使用 Ansible 自动化部署
   - 使用脚本自动化验证
   - 使用监控自动化告警

3. **建立基线并持续对比**
   - 部署后立即建立基线
   - 定期运行性能测试
   - 及时发现性能退化

4. **分阶段部署**
   - 测试 → 金丝雀 → 生产
   - 小批量 → 大批量
   - 始终准备回滚方案

5. **完整的文档**
   - 记录所有配置
   - 保存所有基线
   - 追踪所有变更

6. **持续优化**
   - CPU 优化带来显著性能提升
   - 定期审查配置
   - 关注新版本和新特性

### 下一步行动

**立即执行**:
- [ ] 运行环境检查
- [ ] 配置 Ansible inventory
- [ ] 执行首次部署

**第一周**:
- [ ] 建立性能基线
- [ ] 配置监控
- [ ] 运行完整验证

**第一个月**:
- [ ] 优化配置
- [ ] 建立自动化流程
- [ ] 完善文档

**持续进行**:
- [ ] 监控和告警
- [ ] 定期验证
- [ ] 版本管理
- [ ] 团队培训

---

**本最佳实践指南将持续更新。如有问题或建议，请联系 GPU 集群管理团队。**

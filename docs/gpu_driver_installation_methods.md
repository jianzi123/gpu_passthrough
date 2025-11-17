# GPU 驱动安装方法指南

本文档介绍基于 NVIDIA GPU Operator 和 GPU Driver Container 的多种驱动安装方法。

## 目录

1. [安装方法对比](#安装方法对比)
2. [Native 安装（传统方法）](#native-安装传统方法)
3. [Driver Container 安装](#driver-container-安装)
4. [Precompiled Driver 安装](#precompiled-driver-安装)
5. [方法选择建议](#方法选择建议)
6. [故障排除](#故障排除)

---

## 安装方法对比

### 三种安装方法比较

| 特性 | Native | Driver Container | Precompiled |
|------|--------|------------------|-------------|
| **安装速度** | 慢 (编译) | 中等 | 快 (无编译) |
| **内核解耦** | ❌ 紧耦合 | ✅ 容器化 | ❌ 紧耦合 |
| **版本管理** | 复杂 | 简单 (容器标签) | 中等 |
| **资源消耗** | 高 (编译时) | 低 | 最低 |
| **灵活性** | 低 | 高 | 中 |
| **生产就绪** | ✅ | ✅ | ✅ |
| **Kubernetes 友好** | ❌ | ✅ | ❌ |
| **回滚能力** | 困难 | 简单 (停止容器) | 困难 |
| **适用场景** | 传统部署 | 云原生/K8s | 快速部署 |

### 架构对比

#### Native 安装
```
┌─────────────────┐
│   Application   │
├─────────────────┤
│   CUDA Library  │
├─────────────────┤
│  NVIDIA Driver  │ ← 直接安装到系统
├─────────────────┤
│     Kernel      │
├─────────────────┤
│    Hardware     │
└─────────────────┘
```

#### Driver Container 安装
```
┌─────────────────┐
│   Application   │
├─────────────────┤
│   CUDA Library  │
├─────────────────┤
│ ┌─────────────┐ │
│ │Driver       │ │ ← 容器化驱动
│ │Container    │ │
│ └─────────────┘ │
├─────────────────┤
│     Kernel      │
├─────────────────┤
│    Hardware     │
└─────────────────┘
```

---

## Native 安装（传统方法）

### 概述

传统的驱动安装方式，直接在主机系统上安装 NVIDIA 驱动。

### 优点

- ✅ 成熟稳定，生产环境广泛使用
- ✅ 无需 Docker 等容器运行时
- ✅ 最佳性能（无容器开销）
- ✅ 文档丰富，社区支持好

### 缺点

- ❌ 驱动编译耗时长（每个节点）
- ❌ 内核更新可能破坏驱动
- ❌ 版本管理复杂
- ❌ 回滚困难

### 使用脚本安装

```bash
# 基础安装
sudo ./scripts/install/install_gpu_driver.sh --method native

# 自动检测 GPU 并安装
sudo ./scripts/install/install_gpu_driver.sh \
  --method native \
  --auto-detect

# 指定驱动版本
sudo ./scripts/install/install_gpu_driver.sh \
  --method native \
  --driver-version 535.154.05 \
  --cuda-version 12.2
```

### 使用 Ansible 安装

```yaml
# ansible/playbooks/setup_gpu_baseline.yml

- hosts: gpu_nodes
  become: yes
  roles:
    - role: gpu_baseline
      vars:
        driver_installation_method: native
        auto_detect_cuda_version: true
```

运行：
```bash
cd ansible
ansible-playbook -i inventory/hosts playbooks/setup_gpu_baseline.yml
```

### 验证安装

```bash
# 检查驱动版本
nvidia-smi

# 检查驱动模块
lsmod | grep nvidia

# 检查驱动详细信息
modinfo nvidia

# 测试 CUDA
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv
```

---

## Driver Container 安装

### 概述

基于 NVIDIA GPU Operator 架构，使用容器化方式安装和管理驱动。驱动以容器形式运行，与主机系统解耦。

### 优点

- ✅ 驱动与系统解耦，易于管理
- ✅ 版本管理简单（容器标签）
- ✅ 快速回滚（停止/启动容器）
- ✅ Kubernetes 原生支持
- ✅ 支持多驱动版本并存
- ✅ GPU Operator 标准方法

### 缺点

- ❌ 需要 Docker 或 containerd
- ❌ 额外的容器管理开销
- ❌ 相对较新（2020+）

### 架构组件

Driver Container 包含以下组件：

1. **nvidia-driver container**: 驱动内核模块
2. **systemd service**: 管理驱动容器生命周期
3. **health check**: 驱动健康检查
4. **shared volumes**: 与主机共享 `/run/nvidia`

### 使用脚本安装

```bash
# 基础安装
sudo ./scripts/install/install_gpu_driver.sh \
  --method driver-container

# 指定容器镜像
sudo ./scripts/install/install_gpu_driver.sh \
  --method driver-container \
  --container-image nvcr.io/nvidia/driver \
  --driver-version 535.154.05

# 查看可用的驱动容器镜像
docker search nvcr.io/nvidia/driver
```

### 使用 Ansible 安装

```yaml
# ansible/playbooks/setup_gpu_baseline.yml

- hosts: gpu_nodes
  become: yes
  roles:
    - role: gpu_baseline
      vars:
        driver_installation_method: driver-container
        driver_container_image: "nvcr.io/nvidia/driver"
        driver_container_tag: "535.154.05-ubuntu22.04"
```

运行：
```bash
ansible-playbook -i inventory/hosts playbooks/setup_gpu_baseline.yml
```

### 管理 Driver Container

```bash
# 检查驱动容器状态
systemctl status nvidia-driver

# 查看驱动容器日志
journalctl -u nvidia-driver -f

# 重启驱动容器
systemctl restart nvidia-driver

# 停止驱动容器
systemctl stop nvidia-driver

# 查看容器运行状态
docker ps | grep nvidia-driver

# 进入驱动容器
docker exec -it nvidia-driver bash

# 查看驱动容器配置
cat /etc/nvidia-driver/config

# 健康检查
/usr/local/bin/check-driver-container.sh
```

### 验证安装

```bash
# 检查 nvidia-smi
nvidia-smi

# 检查驱动容器是否运行
docker ps | grep nvidia-driver

# 检查 systemd 服务
systemctl status nvidia-driver

# 验证 GPU 访问
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

### 切换驱动版本

```bash
# 停止当前驱动
systemctl stop nvidia-driver

# 拉取新版本镜像
docker pull nvcr.io/nvidia/driver:550.90.07-ubuntu22.04

# 更新配置
sudo sed -i 's/535.154.05/550.90.07/g' /etc/nvidia-driver/config

# 重新生成 systemd service
# (或手动编辑 /etc/systemd/system/nvidia-driver.service)

# 启动新版本
systemctl daemon-reload
systemctl start nvidia-driver

# 验证
nvidia-smi
```

---

## Precompiled Driver 安装

### 概述

使用预编译的驱动模块，跳过编译步骤，实现快速部署。

### 优点

- ✅ 安装速度最快（无需编译）
- ✅ 节省 CPU 和内存资源
- ✅ 部署一致性好
- ✅ 适合大规模部署

### 缺点

- ❌ 需要匹配的内核版本
- ❌ 内核更新需重新构建
- ❌ 维护预编译库的额外工作

### 构建预编译驱动

```bash
# 为当前内核构建
sudo ./scripts/install/build_precompiled_driver.sh \
  --driver-version 535.154.05

# 为特定内核构建
sudo ./scripts/install/build_precompiled_driver.sh \
  --driver-version 535.154.05 \
  --kernel-version 5.15.0-91-generic

# 在容器中构建（推荐）
sudo ./scripts/install/build_precompiled_driver.sh \
  --driver-version 535.154.05 \
  --container-build

# 指定输出目录
sudo ./scripts/install/build_precompiled_driver.sh \
  --driver-version 535.154.05 \
  --output-dir /opt/precompiled_drivers
```

### 安装预编译驱动

```bash
# 解压预编译包
cd /opt/precompiled_drivers
tar -xzf nvidia-driver-535.154.05-kernel-5.15.0-91-generic.tar.gz
cd nvidia-driver-535.154.05-kernel-5.15.0-91-generic

# 运行安装脚本
sudo ./install.sh

# 验证
nvidia-smi
```

### 使用脚本安装

```bash
sudo ./scripts/install/install_gpu_driver.sh \
  --method precompiled \
  --driver-version 535.154.05 \
  --precompiled
```

### 批量构建多个内核版本

```bash
#!/bin/bash
# 为多个内核版本构建预编译驱动

DRIVER_VERSION="535.154.05"
KERNELS=(
  "5.15.0-91-generic"
  "5.15.0-92-generic"
  "6.2.0-39-generic"
)

for kernel in "${KERNELS[@]}"; do
  echo "Building for kernel: $kernel"
  sudo ./scripts/install/build_precompiled_driver.sh \
    --driver-version $DRIVER_VERSION \
    --kernel-version $kernel \
    --container-build
done

echo "All builds completed"
ls -lh /opt/precompiled_drivers/
```

---

## 方法选择建议

### 场景 1: 传统数据中心 / 物理服务器

**推荐**: Native 安装

**理由**:
- 成熟稳定
- 无需容器运行时
- 最佳性能
- 内核更新不频繁

**配置**:
```yaml
driver_installation_method: native
auto_detect_cuda_version: true
```

### 场景 2: Kubernetes 集群 / 云原生环境

**推荐**: Driver Container

**理由**:
- GPU Operator 标准方法
- 驱动版本管理简单
- 快速回滚
- 支持多驱动版本

**配置**:
```yaml
driver_installation_method: driver-container
driver_container_image: "nvcr.io/nvidia/driver"
driver_container_tag: "535.154.05-ubuntu22.04"
```

### 场景 3: 大规模快速部署 / CI/CD

**推荐**: Precompiled Driver

**理由**:
- 安装速度最快
- 节省资源
- 部署一致性

**前提条件**:
- 内核版本统一
- 有预编译驱动库

**配置**:
```yaml
driver_installation_method: precompiled
use_precompiled: true
```

### 场景 4: 开发/测试环境

**推荐**: Driver Container

**理由**:
- 快速切换驱动版本
- 易于测试不同版本
- 不污染系统

### 场景 5: 混合环境

**推荐**: 根据节点类型选择

```yaml
# GPU 节点组 1: 生产环境
- hosts: gpu_prod
  vars:
    driver_installation_method: native

# GPU 节点组 2: Kubernetes 集群
- hosts: gpu_k8s
  vars:
    driver_installation_method: driver-container

# GPU 节点组 3: 快速部署
- hosts: gpu_batch
  vars:
    driver_installation_method: precompiled
```

---

## 故障排除

### Native 安装问题

#### 问题 1: 编译失败

**症状**: 驱动编译过程中出错

**解决**:
```bash
# 检查内核头文件
dpkg -l | grep linux-headers-$(uname -r)

# 安装缺失的头文件
sudo apt-get install linux-headers-$(uname -r)

# 检查编译工具
gcc --version
make --version

# 查看详细编译日志
sudo cat /var/log/nvidia-installer.log
```

#### 问题 2: nouveau 冲突

**症状**: nouveau 驱动与 NVIDIA 驱动冲突

**解决**:
```bash
# 禁用 nouveau
sudo bash -c "echo blacklist nouveau > /etc/modprobe.d/blacklist-nvidia-nouveau.conf"
sudo bash -c "echo options nouveau modeset=0 >> /etc/modprobe.d/blacklist-nvidia-nouveau.conf"

# 更新 initramfs
sudo update-initramfs -u

# 重启
sudo reboot
```

### Driver Container 问题

#### 问题 1: 容器无法启动

**症状**: Driver container 服务失败

**解决**:
```bash
# 检查 Docker 服务
systemctl status docker

# 查看容器日志
journalctl -u nvidia-driver -n 50

# 手动运行容器测试
docker run --rm --privileged \
  -v /run/nvidia:/run/nvidia:shared \
  nvcr.io/nvidia/driver:535.154.05-ubuntu22.04

# 检查镜像是否存在
docker images | grep nvidia/driver
```

#### 问题 2: nvidia-smi 不可用

**症状**: 容器运行但 nvidia-smi 失败

**解决**:
```bash
# 检查共享卷
ls -la /run/nvidia

# 检查容器内驱动
docker exec nvidia-driver nvidia-smi

# 检查模块加载
lsmod | grep nvidia

# 重启容器
systemctl restart nvidia-driver
```

### Precompiled Driver 问题

#### 问题 1: 内核版本不匹配

**症状**: 预编译驱动与当前内核不匹配

**解决**:
```bash
# 检查当前内核
uname -r

# 重新构建预编译驱动
sudo ./scripts/install/build_precompiled_driver.sh \
  --driver-version 535.154.05 \
  --kernel-version $(uname -r)
```

#### 问题 2: 模块签名问题

**症状**: 安全启动环境下模块加载失败

**解决**:
```bash
# 方法 1: 禁用安全启动（BIOS 设置）

# 方法 2: 签名模块
sudo /usr/src/linux-headers-$(uname -r)/scripts/sign-file \
  sha256 \
  /path/to/private.key \
  /path/to/public.der \
  /lib/modules/$(uname -r)/kernel/drivers/video/nvidia.ko
```

### 通用问题

#### 问题: 检查驱动状态

```bash
# 综合检查脚本
cat > /tmp/check_gpu_driver.sh << 'EOF'
#!/bin/bash

echo "=== GPU Driver Status Check ==="

echo -e "\n1. NVIDIA Driver Version:"
nvidia-smi --query-gpu=driver_version --format=csv,noheader || echo "FAILED"

echo -e "\n2. Kernel Modules:"
lsmod | grep nvidia || echo "No NVIDIA modules loaded"

echo -e "\n3. Device Files:"
ls -la /dev/nvidia* || echo "No device files"

echo -e "\n4. Container Support:"
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi || echo "Container GPU access FAILED"

echo -e "\n5. Installation Method:"
if systemctl is-active nvidia-driver &> /dev/null; then
    echo "Driver Container"
elif [ -f /usr/bin/nvidia-smi ]; then
    echo "Native Installation"
else
    echo "Unknown or not installed"
fi

echo -e "\n=== Check Complete ==="
EOF

chmod +x /tmp/check_gpu_driver.sh
sudo /tmp/check_gpu_driver.sh
```

---

## 参考资料

### NVIDIA 官方文档

- [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/)
- [GPU Driver Container](https://github.com/NVIDIA/gpu-driver-container)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/)
- [Precompiled Drivers](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/precompiled-drivers.html)

### 相关项目

- [NVIDIA DeepOps](https://github.com/NVIDIA/deepops)
- [NVIDIA GPU Operator](https://github.com/NVIDIA/gpu-operator)
- [NVIDIA Device Plugin](https://github.com/NVIDIA/k8s-device-plugin)

### 博客和教程

- [The Real-World Guide to the NVIDIA GPU Operator](https://www.spectrocloud.com/blog/the-real-world-guide-to-the-nvidia-gpu-operator-for-kubernetes-ai)
- [NVIDIA Container Toolkit Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

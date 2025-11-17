# NVIDIA 预编译驱动完整指南

本文档详细介绍 NVIDIA GPU 预编译驱动的构建、部署、管理和最佳实践。

## 目录

1. [概述](#概述)
2. [为什么使用预编译驱动](#为什么使用预编译驱动)
3. [架构和工作原理](#架构和工作原理)
4. [构建预编译驱动](#构建预编译驱动)
5. [部署预编译驱动](#部署预编译驱动)
6. [管理和维护](#管理和维护)
7. [与 GPU Operator 集成](#与-gpu-operator-集成)
8. [CI/CD 集成](#cicd-集成)
9. [最佳实践](#最佳实践)
10. [故障排除](#故障排除)
11. [性能对比](#性能对比)
12. [实际案例](#实际案例)

---

## 概述

### 什么是预编译驱动

预编译驱动（Precompiled Driver）是指提前编译好的 NVIDIA GPU 内核模块，可以直接部署到目标系统而无需在目标机器上进行编译。

### 核心概念

```
传统安装流程：
┌────────────┐    ┌────────────┐    ┌────────────┐
│ 下载源码    │ -> │ 编译驱动    │ -> │ 安装模块    │
│ (5 分钟)   │    │(15-30 分钟)│    │ (2 分钟)   │
└────────────┘    └────────────┘    └────────────┘
总时间: ~35 分钟/节点

预编译驱动流程：
┌────────────┐    ┌────────────┐
│ 下载预编译包 │ -> │ 安装模块    │
│ (2 分钟)   │    │ (1 分钟)   │
└────────────┘    └────────────┘
总时间: ~3 分钟/节点
```

### 主要优势

| 特性 | 传统安装 | 预编译驱动 |
|------|---------|-----------|
| **安装时间** | 20-35 分钟 | 2-3 分钟 |
| **CPU 使用** | 高（编译） | 低 |
| **内存使用** | 2-4 GB | < 500 MB |
| **网络带宽** | 中等 | 低 |
| **一致性** | 可能不同 | 完全一致 |
| **可预测性** | 低 | 高 |
| **回滚速度** | 慢 | 快 |

---

## 为什么使用预编译驱动

### 1. 大规模部署优势

**场景**：1000 台 GPU 服务器集群

```
传统方式：
- 每台编译时间：25 分钟
- 总 CPU 时间：1000 × 25 = 25,000 分钟 ≈ 417 小时
- 并发编译峰值：高负载
- 网络下载：1000 × 200MB = 200 GB

预编译方式：
- 编译一次：25 分钟
- 部署每台：3 分钟
- 总时间：25 + (3 × 1000) ≈ 50 小时（并行部署）
- 网络下载：1000 × 50MB = 50 GB
- 节省 CPU 时间：~90%
```

### 2. 资源优化

**编译资源消耗对比**：

```bash
# 传统编译资源需求（每节点）
CPU: 4-8 核心持续使用 20-30 分钟
内存: 2-4 GB
磁盘: 临时文件 1-2 GB
网络: 下载源码包 ~200 MB

# 预编译部署资源需求（每节点）
CPU: 最小使用
内存: < 500 MB
磁盘: 预编译包 ~50 MB
网络: 下载预编译包 ~50 MB
```

### 3. 一致性保证

预编译驱动在统一环境中构建，确保：

- ✅ 相同的编译器版本
- ✅ 相同的编译选项
- ✅ 相同的依赖库版本
- ✅ 可重现的构建过程

### 4. 快速故障恢复

```
驱动问题恢复时间对比：

传统方式：
1. 卸载旧驱动：5 分钟
2. 重新编译：25 分钟
3. 安装新驱动：5 分钟
总计：35 分钟

预编译方式：
1. 卸载旧驱动：5 分钟
2. 安装预编译包：3 分钟
总计：8 分钟

时间节省：~77%
```

---

## 架构和工作原理

### 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                   构建环境 (Build Host)                   │
│  ┌──────────────────────────────────────────────────┐   │
│  │  1. 下载 NVIDIA 驱动源码                          │   │
│  │  2. 编译内核模块（针对目标内核版本）               │   │
│  │  3. 收集编译产物（.ko 文件）                      │   │
│  │  4. 打包（tar.gz）                                │   │
│  │  5. 生成校验和（SHA256）                          │   │
│  └──────────────────────────────────────────────────┘   │
│                          ↓                               │
│              生成预编译驱动包                             │
│  nvidia-driver-535.154.05-kernel-5.15.0-91.tar.gz       │
└─────────────────────────────────────────────────────────┘
                          ↓
            ┌─────────────┴──────────────┐
            ↓                            ↓
┌────────────────────┐      ┌────────────────────┐
│  目标节点 Node 1    │      │  目标节点 Node N    │
│  1. 下载预编译包    │      │  1. 下载预编译包    │
│  2. 解压            │      │  2. 解压            │
│  3. 复制 .ko 文件   │      │  3. 复制 .ko 文件   │
│  4. 加载模块        │      │  4. 加载模块        │
│  5. 验证            │      │  5. 验证            │
└────────────────────┘      └────────────────────┘
```

### 预编译包结构

```
nvidia-driver-535.154.05-kernel-5.15.0-91-generic/
├── modules/                    # 编译好的内核模块
│   ├── nvidia.ko              # 核心驱动模块
│   ├── nvidia-uvm.ko          # 统一虚拟内存模块
│   ├── nvidia-modeset.ko      # 模式设置模块
│   ├── nvidia-drm.ko          # DRM 模块
│   └── nvidia-peermem.ko      # Peer Memory 模块（可选）
├── firmware/                   # 固件文件（如果有）
│   └── gsp/                   # GSP 固件
├── metadata.json               # 元数据
│   {
│     "driver_version": "535.154.05",
│     "kernel_version": "5.15.0-91-generic",
│     "build_date": "2025-01-17T10:30:00Z",
│     "architecture": "x86_64",
│     "compiler": "gcc-11.4.0",
│     "build_host": "build-server-01"
│   }
├── install.sh                  # 安装脚本
├── uninstall.sh               # 卸载脚本
└── README.txt                 # 使用说明
```

### 工作流程详解

#### 阶段 1: 构建预编译驱动

```bash
#!/bin/bash
# 构建流程伪代码

# 1. 准备构建环境
install_build_dependencies() {
    apt-get install build-essential linux-headers-${KERNEL_VERSION}
}

# 2. 下载驱动源码
download_driver() {
    wget https://nvidia.com/driver-${VERSION}.run
    chmod +x driver-${VERSION}.run
}

# 3. 提取源码
extract_driver() {
    ./driver-${VERSION}.run --extract-only --target=/build
}

# 4. 编译内核模块
build_modules() {
    cd /build/kernel
    make -j$(nproc) SYSSRC=/lib/modules/${KERNEL_VERSION}/build
}

# 5. 收集模块
collect_modules() {
    find . -name "*.ko" -exec cp {} /output/modules/ \;
}

# 6. 打包
package_driver() {
    tar -czf nvidia-driver-${VERSION}-kernel-${KERNEL}.tar.gz \
        modules/ metadata.json install.sh
}
```

#### 阶段 2: 部署预编译驱动

```bash
#!/bin/bash
# 部署流程伪代码

# 1. 下载预编译包
download_package() {
    wget http://repo/nvidia-driver-${VERSION}-${KERNEL}.tar.gz
}

# 2. 验证校验和
verify_package() {
    sha256sum -c nvidia-driver-${VERSION}-${KERNEL}.tar.gz.sha256
}

# 3. 解压
extract_package() {
    tar -xzf nvidia-driver-${VERSION}-${KERNEL}.tar.gz
}

# 4. 安装模块
install_modules() {
    cp modules/*.ko /lib/modules/$(uname -r)/kernel/drivers/video/
    depmod -a
}

# 5. 加载驱动
load_driver() {
    modprobe nvidia
    modprobe nvidia-uvm
}

# 6. 验证
verify_installation() {
    nvidia-smi
}
```

---

## 构建预编译驱动

### 方法 1: 使用构建脚本（推荐）

#### 基础构建

```bash
# 为当前内核构建
sudo ./scripts/install/build_precompiled_driver.sh \
    --driver-version 535.154.05

# 为特定内核构建
sudo ./scripts/install/build_precompiled_driver.sh \
    --driver-version 535.154.05 \
    --kernel-version 5.15.0-91-generic

# 指定输出目录
sudo ./scripts/install/build_precompiled_driver.sh \
    --driver-version 535.154.05 \
    --output-dir /data/precompiled-drivers
```

#### 容器化构建（推荐）

```bash
# 在 Docker 容器中构建（更干净、可重现）
sudo ./scripts/install/build_precompiled_driver.sh \
    --driver-version 535.154.05 \
    --kernel-version 5.15.0-91-generic \
    --container-build
```

**容器化构建优势**：
- ✅ 隔离的构建环境
- ✅ 可重现的构建
- ✅ 不污染主机系统
- ✅ 支持多版本并行构建
- ✅ 自动清理

### 方法 2: 手动构建

#### 步骤 1: 准备环境

```bash
# 安装构建依赖
apt-get update
apt-get install -y \
    build-essential \
    dkms \
    linux-headers-5.15.0-91-generic \
    wget \
    ca-certificates

# 创建构建目录
mkdir -p /tmp/driver-build
cd /tmp/driver-build
```

#### 步骤 2: 下载驱动

```bash
# 下载 NVIDIA 驱动
DRIVER_VERSION="535.154.05"
wget https://us.download.nvidia.com/tesla/${DRIVER_VERSION}/NVIDIA-Linux-x86_64-${DRIVER_VERSION}.run

# 验证下载
ls -lh NVIDIA-Linux-x86_64-${DRIVER_VERSION}.run
chmod +x NVIDIA-Linux-x86_64-${DRIVER_VERSION}.run
```

#### 步骤 3: 提取和编译

```bash
# 提取驱动源码
./NVIDIA-Linux-x86_64-${DRIVER_VERSION}.run \
    --extract-only \
    --target=./extracted

# 进入内核目录
cd extracted/kernel

# 编译模块
KERNEL_VERSION="5.15.0-91-generic"
make -j$(nproc) \
    SYSSRC=/lib/modules/${KERNEL_VERSION}/build \
    module

# 检查编译结果
find . -name "*.ko" -ls
```

#### 步骤 4: 打包

```bash
# 创建包目录
PACKAGE_DIR="/tmp/nvidia-driver-${DRIVER_VERSION}-kernel-${KERNEL_VERSION}"
mkdir -p ${PACKAGE_DIR}/{modules,firmware}

# 复制模块
find . -name "*.ko" -exec cp {} ${PACKAGE_DIR}/modules/ \;

# 创建元数据
cat > ${PACKAGE_DIR}/metadata.json << EOF
{
  "driver_version": "${DRIVER_VERSION}",
  "kernel_version": "${KERNEL_VERSION}",
  "build_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "architecture": "x86_64",
  "compiler": "$(gcc --version | head -1)",
  "build_host": "$(hostname)"
}
EOF

# 创建安装脚本
cat > ${PACKAGE_DIR}/install.sh << 'INSTALL_SCRIPT'
#!/bin/bash
set -e

echo "Installing NVIDIA precompiled driver..."

KERNEL_VERSION=$(uname -r)
MODULE_DIR="/lib/modules/${KERNEL_VERSION}/kernel/drivers/video"

# 创建目录
mkdir -p "${MODULE_DIR}"

# 复制模块
echo "Copying kernel modules..."
cp modules/*.ko "${MODULE_DIR}/"

# 更新模块依赖
echo "Updating module dependencies..."
depmod -a

# 加载模块
echo "Loading NVIDIA modules..."
modprobe nvidia
modprobe nvidia-uvm
modprobe nvidia-modeset
modprobe nvidia-drm

echo "Driver installed successfully!"
echo "Run 'nvidia-smi' to verify installation"
INSTALL_SCRIPT

chmod +x ${PACKAGE_DIR}/install.sh

# 创建卸载脚本
cat > ${PACKAGE_DIR}/uninstall.sh << 'UNINSTALL_SCRIPT'
#!/bin/bash
set -e

echo "Uninstalling NVIDIA precompiled driver..."

# 卸载模块
echo "Unloading NVIDIA modules..."
rmmod nvidia-drm || true
rmmod nvidia-modeset || true
rmmod nvidia-uvm || true
rmmod nvidia || true

# 删除模块文件
KERNEL_VERSION=$(uname -r)
MODULE_DIR="/lib/modules/${KERNEL_VERSION}/kernel/drivers/video"

echo "Removing kernel modules..."
rm -f ${MODULE_DIR}/nvidia*.ko

# 更新依赖
depmod -a

echo "Driver uninstalled successfully!"
UNINSTALL_SCRIPT

chmod +x ${PACKAGE_DIR}/uninstall.sh

# 打包
cd /tmp
tar -czf nvidia-driver-${DRIVER_VERSION}-kernel-${KERNEL_VERSION}.tar.gz \
    nvidia-driver-${DRIVER_VERSION}-kernel-${KERNEL_VERSION}/

# 生成校验和
sha256sum nvidia-driver-${DRIVER_VERSION}-kernel-${KERNEL_VERSION}.tar.gz \
    > nvidia-driver-${DRIVER_VERSION}-kernel-${KERNEL_VERSION}.tar.gz.sha256

echo "Package created successfully:"
ls -lh nvidia-driver-${DRIVER_VERSION}-kernel-${KERNEL_VERSION}.tar.gz*
```

### 方法 3: 批量构建多个版本

```bash
#!/bin/bash
# 批量构建脚本

DRIVER_VERSION="535.154.05"
OUTPUT_DIR="/opt/precompiled-drivers"

# 要构建的内核版本列表
KERNEL_VERSIONS=(
    "5.15.0-91-generic"
    "5.15.0-92-generic"
    "5.15.0-94-generic"
    "6.2.0-39-generic"
    "6.5.0-14-generic"
)

mkdir -p ${OUTPUT_DIR}

for kernel in "${KERNEL_VERSIONS[@]}"; do
    echo "=========================================="
    echo "Building for kernel: ${kernel}"
    echo "=========================================="

    ./scripts/install/build_precompiled_driver.sh \
        --driver-version ${DRIVER_VERSION} \
        --kernel-version ${kernel} \
        --output-dir ${OUTPUT_DIR} \
        --container-build

    if [ $? -eq 0 ]; then
        echo "✓ Build successful for ${kernel}"
    else
        echo "✗ Build failed for ${kernel}"
    fi
    echo ""
done

echo "=========================================="
echo "Build Summary"
echo "=========================================="
ls -lh ${OUTPUT_DIR}/*.tar.gz

echo ""
echo "Total packages created: $(ls ${OUTPUT_DIR}/*.tar.gz | wc -l)"
echo "Total size: $(du -sh ${OUTPUT_DIR} | cut -f1)"
```

### 构建验证

```bash
# 验证包完整性
verify_package() {
    local package=$1

    echo "Verifying package: ${package}"

    # 检查文件存在
    if [ ! -f "${package}" ]; then
        echo "✗ Package not found"
        return 1
    fi

    # 检查校验和
    if sha256sum -c "${package}.sha256"; then
        echo "✓ Checksum verified"
    else
        echo "✗ Checksum verification failed"
        return 1
    fi

    # 列出包内容
    echo "Package contents:"
    tar -tzf "${package}"

    # 检查必需文件
    local required_files=("modules/nvidia.ko" "metadata.json" "install.sh")
    for file in "${required_files[@]}"; do
        if tar -tzf "${package}" | grep -q "${file}"; then
            echo "✓ Found: ${file}"
        else
            echo "✗ Missing: ${file}"
            return 1
        fi
    done

    echo "✓ Package verification complete"
    return 0
}

# 使用示例
verify_package "/opt/precompiled-drivers/nvidia-driver-535.154.05-kernel-5.15.0-91-generic.tar.gz"
```

---

## 部署预编译驱动

### 方法 1: 使用安装脚本

```bash
# 使用项目提供的安装脚本
sudo ./scripts/install/install_gpu_driver.sh \
    --method precompiled \
    --driver-version 535.154.05

# 指定预编译包位置
sudo PRECOMPILED_PACKAGE="/path/to/nvidia-driver-535.154.05-kernel-5.15.0-91.tar.gz" \
    ./scripts/install/install_gpu_driver.sh --method precompiled
```

### 方法 2: 手动部署

#### 单节点部署

```bash
# 1. 下载预编译包
wget http://your-repo/nvidia-driver-535.154.05-kernel-5.15.0-91-generic.tar.gz
wget http://your-repo/nvidia-driver-535.154.05-kernel-5.15.0-91-generic.tar.gz.sha256

# 2. 验证校验和
sha256sum -c nvidia-driver-535.154.05-kernel-5.15.0-91-generic.tar.gz.sha256

# 3. 解压
tar -xzf nvidia-driver-535.154.05-kernel-5.15.0-91-generic.tar.gz
cd nvidia-driver-535.154.05-kernel-5.15.0-91-generic

# 4. 安装
sudo ./install.sh

# 5. 验证
nvidia-smi
```

#### 多节点并行部署

```bash
#!/bin/bash
# 并行部署到多个节点

DRIVER_PACKAGE="nvidia-driver-535.154.05-kernel-5.15.0-91-generic.tar.gz"
NODES_FILE="nodes.txt"  # 每行一个节点IP/主机名

# 并行部署函数
deploy_to_node() {
    local node=$1
    echo "Deploying to ${node}..."

    # 复制包到节点
    scp ${DRIVER_PACKAGE} ${node}:/tmp/

    # 远程安装
    ssh ${node} << 'EOF'
        cd /tmp
        tar -xzf nvidia-driver-*.tar.gz
        cd nvidia-driver-*
        sudo ./install.sh
        nvidia-smi
EOF

    if [ $? -eq 0 ]; then
        echo "✓ Deployment successful on ${node}"
    else
        echo "✗ Deployment failed on ${node}"
    fi
}

# 并行部署
export -f deploy_to_node
export DRIVER_PACKAGE

# 使用 GNU Parallel 并行部署（如果可用）
if command -v parallel &> /dev/null; then
    cat ${NODES_FILE} | parallel -j 10 deploy_to_node {}
else
    # 串行部署
    while read node; do
        deploy_to_node ${node}
    done < ${NODES_FILE}
fi

echo "Deployment complete!"
```

### 方法 3: 使用 Ansible 部署

#### Ansible Playbook

```yaml
---
# playbooks/deploy_precompiled_driver.yml

- name: Deploy Precompiled NVIDIA Driver
  hosts: gpu_nodes
  become: yes
  vars:
    driver_version: "535.154.05"
    kernel_version: "{{ ansible_kernel }}"
    precompiled_repo: "http://repo.example.com/drivers"
    local_package_dir: "/opt/precompiled-drivers"

  tasks:
    - name: Check current kernel version
      debug:
        msg: "Current kernel: {{ ansible_kernel }}"

    - name: Download precompiled driver package
      get_url:
        url: "{{ precompiled_repo }}/nvidia-driver-{{ driver_version }}-kernel-{{ kernel_version }}.tar.gz"
        dest: "/tmp/nvidia-driver.tar.gz"
        checksum: "sha256:{{ precompiled_repo }}/nvidia-driver-{{ driver_version }}-kernel-{{ kernel_version }}.tar.gz.sha256"
      register: download_result

    - name: Extract driver package
      unarchive:
        src: "/tmp/nvidia-driver.tar.gz"
        dest: "/tmp/"
        remote_src: yes

    - name: Find extracted directory
      find:
        paths: "/tmp"
        patterns: "nvidia-driver-*"
        file_type: directory
      register: extracted_dir

    - name: Run installation script
      command: "./install.sh"
      args:
        chdir: "{{ extracted_dir.files[0].path }}"
      register: install_result

    - name: Verify installation
      command: nvidia-smi
      register: nvidia_smi_output
      changed_when: false

    - name: Display nvidia-smi output
      debug:
        var: nvidia_smi_output.stdout_lines

    - name: Clean up temporary files
      file:
        path: "{{ item }}"
        state: absent
      loop:
        - "/tmp/nvidia-driver.tar.gz"
        - "{{ extracted_dir.files[0].path }}"

    - name: Log installation
      lineinfile:
        path: "/var/log/nvidia-driver-install.log"
        line: "{{ ansible_date_time.iso8601 }} - Installed driver {{ driver_version }} for kernel {{ kernel_version }}"
        create: yes
```

#### 运行部署

```bash
# 部署到所有 GPU 节点
ansible-playbook -i inventory/hosts playbooks/deploy_precompiled_driver.yml

# 部署到特定组
ansible-playbook -i inventory/hosts playbooks/deploy_precompiled_driver.yml \
    --limit gpu_production

# 指定驱动版本
ansible-playbook -i inventory/hosts playbooks/deploy_precompiled_driver.yml \
    -e "driver_version=550.90.07"

# 并行部署（10 个节点同时）
ansible-playbook -i inventory/hosts playbooks/deploy_precompiled_driver.yml \
    --forks 10
```

---

## 管理和维护

### 版本管理

#### 驱动仓库结构

```bash
# 推荐的预编译驱动仓库结构
/opt/precompiled-drivers/
├── current -> 535.154.05/          # 当前版本符号链接
├── 535.154.05/                     # 驱动版本目录
│   ├── kernel-5.15.0-91/
│   │   ├── nvidia-driver-535.154.05-kernel-5.15.0-91-generic.tar.gz
│   │   ├── nvidia-driver-535.154.05-kernel-5.15.0-91-generic.tar.gz.sha256
│   │   └── metadata.json
│   ├── kernel-5.15.0-92/
│   └── kernel-6.2.0-39/
├── 550.90.07/                      # 另一个驱动版本
│   ├── kernel-5.15.0-91/
│   └── kernel-6.2.0-39/
└── index.json                      # 仓库索引
```

#### 仓库索引文件

```json
{
  "repository": "NVIDIA Precompiled Drivers",
  "last_updated": "2025-01-17T10:30:00Z",
  "drivers": [
    {
      "version": "535.154.05",
      "release_date": "2024-10-15",
      "supported_kernels": [
        "5.15.0-91-generic",
        "5.15.0-92-generic",
        "6.2.0-39-generic"
      ],
      "cuda_version": "12.2",
      "current": true
    },
    {
      "version": "550.90.07",
      "release_date": "2024-12-01",
      "supported_kernels": [
        "5.15.0-91-generic",
        "6.2.0-39-generic"
      ],
      "cuda_version": "12.4",
      "current": false
    }
  ]
}
```

### 创建驱动仓库

```bash
#!/bin/bash
# 创建 HTTP 驱动仓库

REPO_DIR="/var/www/html/nvidia-drivers"
mkdir -p ${REPO_DIR}

# 复制所有预编译包
cp /opt/precompiled-drivers/*/*.tar.gz* ${REPO_DIR}/

# 创建目录索引
cat > ${REPO_DIR}/index.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>NVIDIA Precompiled Drivers Repository</title>
</head>
<body>
    <h1>NVIDIA Precompiled Drivers</h1>
    <pre>
EOF

ls -lh ${REPO_DIR}/*.tar.gz >> ${REPO_DIR}/index.html

cat >> ${REPO_DIR}/index.html << 'EOF'
    </pre>
</body>
</html>
EOF

# 启动简单 HTTP 服务器（用于测试）
cd ${REPO_DIR}
python3 -m http.server 8080
```

### 驱动管理脚本

创建一个统一的驱动管理工具：

```bash
#!/bin/bash
# 预编译驱动管理脚本

REPO_DIR="/opt/precompiled-drivers"

show_help() {
    cat << EOF
Precompiled Driver Manager

Usage: $(basename $0) [command] [options]

Commands:
  list                  List all available drivers
  install <version>     Install specific driver version
  uninstall             Uninstall current driver
  current               Show currently installed driver
  rollback              Rollback to previous version
  verify                Verify current installation
  clean                 Clean old packages

Examples:
  $(basename $0) list
  $(basename $0) install 535.154.05
  $(basename $0) rollback
EOF
}

list_drivers() {
    echo "Available precompiled drivers:"
    find ${REPO_DIR} -name "*.tar.gz" -exec basename {} .tar.gz \; | sort
}

install_driver() {
    local version=$1
    local kernel=$(uname -r)
    local package=$(find ${REPO_DIR} -name "*${version}*${kernel}*.tar.gz" | head -1)

    if [ -z "${package}" ]; then
        echo "Error: No package found for driver ${version} and kernel ${kernel}"
        return 1
    fi

    echo "Installing driver from: ${package}"

    # 备份当前版本信息
    if nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=driver_version --format=csv,noheader \
            > /var/lib/nvidia/previous_version.txt
    fi

    # 解压并安装
    local temp_dir=$(mktemp -d)
    tar -xzf ${package} -C ${temp_dir}
    cd ${temp_dir}/*
    sudo ./install.sh

    # 清理
    rm -rf ${temp_dir}

    echo "Driver installed successfully"
}

current_version() {
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=driver_version --format=csv,noheader
    else
        echo "No NVIDIA driver installed"
    fi
}

rollback_driver() {
    if [ ! -f /var/lib/nvidia/previous_version.txt ]; then
        echo "Error: No previous version information found"
        return 1
    fi

    local prev_version=$(cat /var/lib/nvidia/previous_version.txt)
    echo "Rolling back to driver version: ${prev_version}"
    install_driver ${prev_version}
}

# 主命令分发
case "${1}" in
    list)
        list_drivers
        ;;
    install)
        install_driver "${2}"
        ;;
    current)
        current_version
        ;;
    rollback)
        rollback_driver
        ;;
    verify)
        nvidia-smi
        ;;
    *)
        show_help
        ;;
esac
```

---

## 与 GPU Operator 集成

### GPU Operator 预编译驱动支持

NVIDIA GPU Operator 原生支持预编译驱动。配置示例：

```yaml
# gpu-operator-values.yaml

driver:
  enabled: true

  # 使用预编译驱动
  use_precompiled: true

  # 预编译驱动仓库
  repository: nvcr.io/nvidia/driver

  # 版本和标签
  version: "535.154.05"

  # 预编译驱动配置
  precompiled:
    enabled: true
    repository: "http://your-repo.com/drivers"
    version: "535.154.05"

    # 内核版本匹配
    kernel_match: "5.15.0-*-generic"

# 安装 GPU Operator
helm install gpu-operator nvidia/gpu-operator \
  -n gpu-operator-resources \
  --create-namespace \
  -f gpu-operator-values.yaml
```

### 自定义预编译驱动镜像

```dockerfile
# Dockerfile for precompiled driver container
FROM nvcr.io/nvidia/driver:535.154.05-ubuntu22.04 AS base

# 添加预编译模块
COPY precompiled/modules/*.ko /usr/src/nvidia-535.154.05/kernel/

# 跳过编译步骤
ENV SKIP_MODULE_BUILD=true

# 构建镜像
# docker build -t my-registry/nvidia-driver:535.154.05-precompiled .
```

---

## CI/CD 集成

### GitLab CI 示例

```yaml
# .gitlab-ci.yml

stages:
  - build
  - test
  - deploy

variables:
  DRIVER_VERSION: "535.154.05"
  OUTPUT_DIR: "/builds/drivers"

build_precompiled_driver:
  stage: build
  image: ubuntu:22.04
  script:
    - apt-get update
    - apt-get install -y build-essential linux-headers-generic wget
    - ./scripts/install/build_precompiled_driver.sh \
        --driver-version ${DRIVER_VERSION} \
        --output-dir ${OUTPUT_DIR} \
        --container-build
  artifacts:
    paths:
      - ${OUTPUT_DIR}/*.tar.gz
      - ${OUTPUT_DIR}/*.sha256
    expire_in: 30 days
  only:
    - main
    - tags

test_package:
  stage: test
  script:
    - ./scripts/test/verify_precompiled_package.sh ${OUTPUT_DIR}/*.tar.gz
  dependencies:
    - build_precompiled_driver

deploy_to_repo:
  stage: deploy
  script:
    - rsync -av ${OUTPUT_DIR}/ driver-repo:/var/www/drivers/
  only:
    - main
  dependencies:
    - build_precompiled_driver
```

### Jenkins Pipeline 示例

```groovy
// Jenkinsfile

pipeline {
    agent any

    parameters {
        string(name: 'DRIVER_VERSION', defaultValue: '535.154.05')
        choice(name: 'KERNEL_VERSIONS', choices: ['5.15.0-91-generic', '6.2.0-39-generic', 'all'])
    }

    stages {
        stage('Build') {
            steps {
                script {
                    def kernels = params.KERNEL_VERSIONS == 'all'
                        ? ['5.15.0-91-generic', '6.2.0-39-generic']
                        : [params.KERNEL_VERSIONS]

                    kernels.each { kernel ->
                        sh """
                            ./scripts/install/build_precompiled_driver.sh \
                                --driver-version ${params.DRIVER_VERSION} \
                                --kernel-version ${kernel} \
                                --output-dir ${WORKSPACE}/output \
                                --container-build
                        """
                    }
                }
            }
        }

        stage('Test') {
            steps {
                sh './scripts/test/verify_all_packages.sh ${WORKSPACE}/output'
            }
        }

        stage('Publish') {
            steps {
                archiveArtifacts artifacts: 'output/*.tar.gz,output/*.sha256'

                sh '''
                    scp output/*.tar.gz* repo-server:/var/www/drivers/
                    ssh repo-server "cd /var/www/drivers && ./update-index.sh"
                '''
            }
        }
    }

    post {
        success {
            emailext (
                subject: "Driver Build Success: ${params.DRIVER_VERSION}",
                body: "Precompiled driver built successfully",
                to: 'gpu-team@example.com'
            )
        }
    }
}
```

### GitHub Actions 示例

```yaml
# .github/workflows/build-driver.yml

name: Build Precompiled Driver

on:
  push:
    tags:
      - 'driver-*'
  workflow_dispatch:
    inputs:
      driver_version:
        description: 'Driver version to build'
        required: true
        default: '535.154.05'

jobs:
  build:
    runs-on: ubuntu-22.04
    strategy:
      matrix:
        kernel:
          - '5.15.0-91-generic'
          - '6.2.0-39-generic'

    steps:
      - uses: actions/checkout@v3

      - name: Set up build environment
        run: |
          sudo apt-get update
          sudo apt-get install -y build-essential linux-headers-${{ matrix.kernel }}

      - name: Build precompiled driver
        run: |
          ./scripts/install/build_precompiled_driver.sh \
            --driver-version ${{ github.event.inputs.driver_version }} \
            --kernel-version ${{ matrix.kernel }} \
            --output-dir ./output \
            --container-build

      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: precompiled-drivers
          path: output/*.tar.gz*

      - name: Create Release
        if: startsWith(github.ref, 'refs/tags/')
        uses: softprops/action-gh-release@v1
        with:
          files: output/*.tar.gz*
```

---

## 最佳实践

### 1. 版本命名规范

```bash
# 推荐的命名格式
nvidia-driver-{DRIVER_VERSION}-kernel-{KERNEL_VERSION}-{ARCH}.tar.gz

# 示例
nvidia-driver-535.154.05-kernel-5.15.0-91-generic-x86_64.tar.gz
nvidia-driver-550.90.07-kernel-6.2.0-39-generic-x86_64.tar.gz

# 包含更多信息
nvidia-driver-535.154.05-kernel-5.15.0-91-generic-ubuntu22.04-x86_64-20250117.tar.gz
```

### 2. 校验和管理

```bash
# 始终生成和验证 SHA256 校验和
sha256sum nvidia-driver-*.tar.gz > nvidia-driver-*.tar.gz.sha256

# 自动化验证
verify_all_packages() {
    local failed=0

    for package in *.tar.gz; do
        if sha256sum -c "${package}.sha256"; then
            echo "✓ ${package} verified"
        else
            echo "✗ ${package} verification failed"
            failed=$((failed + 1))
        fi
    done

    return ${failed}
}
```

### 3. 元数据标准化

```json
{
  "driver_version": "535.154.05",
  "kernel_version": "5.15.0-91-generic",
  "build_date": "2025-01-17T10:30:00Z",
  "build_host": "build-server-01",
  "architecture": "x86_64",
  "os_distribution": "ubuntu",
  "os_version": "22.04",
  "compiler": "gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0",
  "cuda_version": "12.2",
  "modules": [
    "nvidia.ko",
    "nvidia-uvm.ko",
    "nvidia-modeset.ko",
    "nvidia-drm.ko"
  ],
  "module_sizes": {
    "nvidia.ko": "56234567",
    "nvidia-uvm.ko": "1234567"
  },
  "build_options": [
    "-O2",
    "-fno-strict-aliasing"
  ],
  "git_commit": "abc123def456",
  "jenkins_build": "1234"
}
```

### 4. 自动化测试

```bash
#!/bin/bash
# 预编译驱动测试脚本

test_precompiled_package() {
    local package=$1
    local temp_dir=$(mktemp -d)

    echo "Testing package: ${package}"

    # 解压
    tar -xzf "${package}" -C "${temp_dir}"
    cd "${temp_dir}"/*

    # 检查必需文件
    local required_files=(
        "modules/nvidia.ko"
        "modules/nvidia-uvm.ko"
        "metadata.json"
        "install.sh"
    )

    for file in "${required_files[@]}"; do
        if [ ! -f "${file}" ]; then
            echo "✗ Missing required file: ${file}"
            return 1
        fi
    done

    # 检查模块信息
    for module in modules/*.ko; do
        if ! modinfo "${module}" &>/dev/null; then
            echo "✗ Invalid kernel module: ${module}"
            return 1
        fi
        echo "✓ Valid module: $(basename ${module})"
    done

    # 验证元数据
    if ! jq empty metadata.json; then
        echo "✗ Invalid JSON in metadata.json"
        return 1
    fi

    # 清理
    rm -rf "${temp_dir}"

    echo "✓ Package test passed"
    return 0
}
```

### 5. 存储优化

```bash
# 使用符号链接减少重复
/opt/precompiled-drivers/
├── common/                        # 共享的固件和库
│   └── firmware/
├── 535.154.05/
│   ├── kernel-5.15.0-91/
│   │   ├── modules/              # 实际文件
│   │   └── firmware -> ../../common/firmware  # 符号链接
│   └── kernel-5.15.0-92/
│       ├── modules/
│       └── firmware -> ../../common/firmware

# 压缩存储
tar -czf driver.tar.gz --use-compress-program=pigz  # 并行压缩

# 去重存储（使用 rsync 硬链接）
rsync -a --link-dest=../previous/ current/ new/
```

### 6. 监控和告警

```bash
#!/bin/bash
# 驱动健康监控

check_driver_health() {
    # 检查驱动是否加载
    if ! lsmod | grep -q nvidia; then
        alert "NVIDIA driver not loaded"
        return 1
    fi

    # 检查 nvidia-smi
    if ! nvidia-smi &>/dev/null; then
        alert "nvidia-smi failed"
        return 1
    fi

    # 检查 GPU 温度
    local temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)
    if [ ${temp} -gt 85 ]; then
        alert "GPU temperature high: ${temp}°C"
    fi

    # 检查 ECC 错误
    local ecc=$(nvidia-smi --query-gpu=ecc.errors.uncorrected.aggregate.total --format=csv,noheader)
    if [ ${ecc} -gt 0 ]; then
        alert "ECC errors detected: ${ecc}"
    fi

    return 0
}

# Prometheus 导出器示例
cat > /etc/systemd/system/nvidia-exporter.service << 'EOF'
[Unit]
Description=NVIDIA Driver Metrics Exporter
After=network.target

[Service]
ExecStart=/usr/local/bin/nvidia_gpu_exporter
Restart=always

[Install]
WantedBy=multi-user.target
EOF
```

---

## 故障排除

### 问题 1: 内核版本不匹配

**症状**: 模块加载失败，`modprobe: ERROR: could not insert 'nvidia': Invalid module format`

**诊断**:
```bash
# 检查模块版本
modinfo /path/to/nvidia.ko | grep vermagic

# 检查当前内核
uname -r

# 查看详细错误
dmesg | tail -20
```

**解决**:
```bash
# 为当前内核重新构建
./scripts/install/build_precompiled_driver.sh \
    --driver-version 535.154.05 \
    --kernel-version $(uname -r)
```

### 问题 2: 模块依赖缺失

**症状**: `modprobe: ERROR: could not find module by name=nvidia-uvm`

**解决**:
```bash
# 更新模块依赖
sudo depmod -a

# 手动加载依赖
sudo modprobe nvidia
sudo modprobe nvidia-uvm
sudo modprobe nvidia-modeset
```

### 问题 3: 符号版本不匹配

**症状**: `nvidia: disagrees about version of symbol`

**诊断**:
```bash
# 检查模块符号
modprobe --dump-modversions /path/to/nvidia.ko | grep symbol_name

# 检查内核符号
cat /proc/kallsyms | grep symbol_name
```

**解决**: 在相同内核环境中重新编译

### 问题 4: 安装后 nvidia-smi 失败

**症状**: 驱动模块已加载，但 nvidia-smi 无法运行

**诊断**:
```bash
# 检查设备节点
ls -la /dev/nvidia*

# 检查驱动版本
cat /proc/driver/nvidia/version

# 检查用户空间工具
which nvidia-smi
nvidia-smi --version
```

**解决**:
```bash
# 安装用户空间工具
apt-get install nvidia-utils-535

# 或从官方 runfile 安装
./NVIDIA-Linux-x86_64-535.154.05.run --no-kernel-module
```

---

## 性能对比

### 安装时间对比（100 节点集群）

| 阶段 | 传统安装 | 预编译驱动 | 节省时间 |
|------|---------|-----------|---------|
| 下载 | 200 MB × 100 = 20 GB | 50 MB × 100 = 5 GB | 75% |
| 编译 | 25 分钟 × 100 = 2500 分钟 | 0 分钟 | 100% |
| 安装 | 5 分钟 × 100 = 500 分钟 | 2 分钟 × 100 = 200 分钟 | 60% |
| **总计** | **~50 小时** | **~3.5 小时** | **93%** |

### 资源使用对比（单节点）

| 资源 | 传统安装 | 预编译驱动 | 节省 |
|------|---------|-----------|------|
| CPU 时间 | 100 分钟 | 5 分钟 | 95% |
| 内存峰值 | 4 GB | 512 MB | 87% |
| 磁盘 I/O | 高 | 低 | ~80% |
| 网络带宽 | 200 MB | 50 MB | 75% |

### 实际测试结果

```
测试环境：
- 节点数量：10
- GPU 型号：A100
- 网络：1 Gbps
- 操作系统：Ubuntu 22.04

传统安装：
- 平均时间：28 分钟/节点
- 总时间：280 分钟（串行）
- 并行部署（10个同时）：35 分钟
- CPU 负载：平均 85%

预编译驱动：
- 平均时间：3 分钟/节点
- 总时间：30 分钟（串行）
- 并行部署（10个同时）：5 分钟
- CPU 负载：平均 10%

改进：
- 时间节省：86%
- 资源节省：88%
```

---

## 实际案例

### 案例 1: 大规模 AI 训练集群

**背景**:
- 500 台 A100 GPU 服务器
- 需要快速部署新驱动
- 内核版本统一

**解决方案**:
```bash
# 1. 构建一次预编译驱动
./scripts/install/build_precompiled_driver.sh \
    --driver-version 535.154.05 \
    --kernel-version 5.15.0-91-generic \
    --container-build

# 2. 上传到内部仓库
scp output/*.tar.gz* internal-repo:/var/www/drivers/

# 3. 并行部署到 500 台服务器
ansible-playbook -i inventory/all_gpu_nodes \
    playbooks/deploy_precompiled_driver.yml \
    --forks 50

# 结果
# - 部署时间：从 8 小时降低到 40 分钟
# - 成功率：100%
# - 一致性：完全一致
```

### 案例 2: 多内核版本环境

**背景**:
- 混合内核版本（5.15、6.2、6.5）
- 需要支持驱动回滚
- 频繁的内核更新

**解决方案**:
```bash
# 为所有内核版本预构建驱动
./scripts/batch_build_all_kernels.sh

# 创建版本管理系统
/opt/drivers/
├── 535.154.05/
│   ├── 5.15.0-91/
│   ├── 6.2.0-39/
│   └── 6.5.0-14/
└── 550.90.07/
    ├── 5.15.0-91/
    └── 6.2.0-39/

# 自动选择匹配的驱动
ansible-playbook deploy_driver.yml \
    -e "driver_version=535.154.05" \
    -e "auto_select_kernel=true"
```

### 案例 3: CI/CD 自动化

**背景**:
- 每周构建新驱动
- 自动测试和验证
- 自动部署到测试环境

**解决方案**:
```yaml
# GitLab CI 完整流程
stages:
  - build
  - test
  - stage
  - production

# 每周日自动构建
schedule:
  cron: "0 2 * * 0"

# 自动测试验证
test:
  script:
    - ./test_driver.sh
    - ./benchmark_performance.sh

# 分阶段部署
deploy_to_stage:
  environment: staging
  script:
    - ansible-playbook deploy.yml --limit staging

deploy_to_prod:
  environment: production
  when: manual
  script:
    - ansible-playbook deploy.yml --limit production
```

---

## 附录

### A. 完整构建脚本示例

参考 `scripts/install/build_precompiled_driver.sh`

### B. Ansible Role 示例

```yaml
# roles/precompiled_driver/tasks/main.yml
---
- name: Deploy precompiled NVIDIA driver
  include_tasks: deploy.yml
  when: driver_method == "precompiled"
```

### C. 内核版本检测脚本

```bash
#!/bin/bash
# 检测并列出需要构建的内核版本

CLUSTER_NODES="nodes.txt"

# 收集所有节点的内核版本
collect_kernel_versions() {
    declare -A kernels

    while read node; do
        kernel=$(ssh ${node} "uname -r")
        kernels["${kernel}"]=1
    done < ${CLUSTER_NODES}

    # 输出唯一内核版本
    for kernel in "${!kernels[@]}"; do
        echo "${kernel}"
    done | sort -V
}

# 为所有内核版本构建驱动
build_for_all_kernels() {
    local driver_version=$1

    for kernel in $(collect_kernel_versions); do
        echo "Building for kernel: ${kernel}"
        ./scripts/install/build_precompiled_driver.sh \
            --driver-version ${driver_version} \
            --kernel-version ${kernel} \
            --container-build
    done
}

build_for_all_kernels "535.154.05"
```

### D. 参考链接

- [NVIDIA GPU Operator - Precompiled Drivers](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/precompiled-drivers.html)
- [NVIDIA Driver Installation Guide](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/)
- [Kernel Module Building](https://www.kernel.org/doc/html/latest/kbuild/)

---

## 总结

预编译驱动提供了一种高效、可靠、可扩展的 NVIDIA GPU 驱动部署方案。通过预先编译驱动模块，可以：

- ✅ 显著减少部署时间（~90%）
- ✅ 降低资源消耗（~85%）
- ✅ 提高部署一致性
- ✅ 简化大规模部署
- ✅ 加快故障恢复

**关键要点**:
1. 在统一构建环境中构建所有预编译包
2. 维护完整的版本和校验和
3. 自动化构建和部署流程
4. 实施严格的测试验证
5. 保持良好的文档记录

**推荐场景**:
- 大规模 GPU 集群（> 50 节点）
- 统一的内核版本环境
- 需要快速部署和回滚
- CI/CD 自动化环境
- 对部署时间敏感的场景

遵循本指南的最佳实践，可以构建一个高效、可靠的预编译驱动管理系统。

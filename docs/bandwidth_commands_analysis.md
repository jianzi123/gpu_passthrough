# Bandwidth 测试命令分析报告

## 问题背景

在 bandwidth 相关测试脚本中，使用了多个专用测试工具，这些工具**并非标准的开源命令**，需要单独安装。

## 使用的命令工具清单

### 1. CUDA Samples 工具（需单独编译）

#### `bandwidthTest`
- **功能**: 测试 PCIe 带宽（Host-to-Device, Device-to-Host）
- **来源**: CUDA Samples
- **是否开源**: ✅ 开源（NVIDIA CUDA Samples）
- **获取方式**: 需要编译 CUDA Samples
- **位置**:
  ```bash
  /usr/local/cuda/extras/demo_suite/bandwidthTest
  /usr/local/cuda/samples/bin/bandwidthTest
  ```

#### `p2pBandwidthLatencyTest`
- **功能**: 测试 GPU-to-GPU P2P 带宽和延迟
- **来源**: CUDA Samples
- **是否开源**: ✅ 开源（NVIDIA CUDA Samples）
- **获取方式**: 需要编译 CUDA Samples
- **位置**:
  ```bash
  /usr/local/cuda/extras/demo_suite/p2pBandwidthLatencyTest
  /usr/local/cuda/samples/bin/p2pBandwidthLatencyTest
  ```

### 2. NVIDIA 官方工具

#### `nvbandwidth`
- **功能**: NVIDIA 官方带宽测试工具（较新）
- **来源**: NVIDIA 官方仓库
- **是否开源**: ✅ 开源
- **GitHub**: https://github.com/NVIDIA/nvbandwidth
- **特点**: 比 CUDA Samples 更全面、更现代的带宽测试工具
- **支持测试**:
  - Host-to-Device
  - Device-to-Host
  - Device-to-Device (NVLink)
  - 各种内存拷贝模式

### 3. RDMA/InfiniBand 工具

#### `ib_write_bw`
- **功能**: RDMA 写带宽测试
- **来源**: perftest 包
- **是否开源**: ✅ 开源
- **GitHub**: https://github.com/linux-rdma/perftest
- **包名**: `perftest`
- **安装方式**:
  ```bash
  # Ubuntu/Debian
  apt-get install perftest

  # RHEL/CentOS
  yum install perftest
  ```

#### 其他 perftest 工具
- `ib_read_bw` - RDMA 读带宽测试
- `ib_send_bw` - RDMA 发送带宽测试
- `ib_write_lat` - RDMA 写延迟测试

### 4. 标准系统工具（通常已安装）

#### `ibstat`
- **功能**: 查看 InfiniBand 设备状态
- **来源**: infiniband-diags 包
- **安装方式**:
  ```bash
  # Ubuntu/Debian
  apt-get install infiniband-diags

  # RHEL/CentOS
  yum install infiniband-diags
  ```

## 安装状态检查

```bash
# 检查当前系统上哪些工具可用
Command                      Status    Source
-----------------------------------------------------------
nvbandwidth                  ❌ 未安装  需编译安装
bandwidthTest                ❌ 未安装  需编译 CUDA Samples
p2pBandwidthLatencyTest      ❌ 未安装  需编译 CUDA Samples
ib_write_bw                  ❌ 未安装  需安装 perftest
ibstat                       ❌ 未安装  需安装 infiniband-diags
```

## 为什么这些命令不存在？

### 原因分析

1. **CUDA Samples 不再默认安装**
   - CUDA 11.6+ 版本后，CUDA Samples 不再随 CUDA Toolkit 一起安装
   - 需要从 GitHub 单独下载并编译

2. **nvbandwidth 是独立项目**
   - 这是 NVIDIA 较新的工具（2023年发布）
   - 需要单独下载和编译

3. **RDMA 工具是可选组件**
   - 只有配置了 InfiniBand/RoCE 网络的系统才需要
   - 不是所有系统都有这些设备

4. **Ansible 自动化安装脚本的作用**
   - 项目中的 Ansible playbooks 就是为了解决这个问题
   - 自动下载、编译、安装所有这些工具

## 如何安装这些工具

### 方法 1: 使用项目的 Ansible 脚本（推荐）

```bash
cd ansible
ansible-playbook playbooks/install_benchmark_tools.yml
```

这个 playbook 会自动安装：
- CUDA Samples（编译 bandwidthTest, p2pBandwidthLatencyTest）
- nvbandwidth（从源码编译）
- perftest（RDMA 测试工具）
- NCCL tests

### 方法 2: 手动安装各个组件

#### 安装 CUDA Samples

```bash
# 克隆 CUDA Samples
git clone https://github.com/NVIDIA/cuda-samples.git
cd cuda-samples

# 编译带宽测试工具
cd Samples/1_Utilities/bandwidthTest
make
sudo cp bandwidthTest /usr/local/bin/

cd ../p2pBandwidthLatencyTest
make
sudo cp p2pBandwidthLatencyTest /usr/local/bin/
```

#### 安装 nvbandwidth

```bash
# 克隆 nvbandwidth
git clone https://github.com/NVIDIA/nvbandwidth.git
cd nvbandwidth

# 编译
make

# 安装
sudo cp nvbandwidth /usr/local/bin/
```

#### 安装 RDMA 工具

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    perftest \
    infiniband-diags \
    libibverbs-dev \
    rdma-core

# RHEL/CentOS/Rocky
sudo yum install -y \
    perftest \
    infiniband-diags \
    libibverbs-devel \
    rdma-core-devel
```

### 方法 3: 使用 Ansible role

```bash
# 只安装 CUDA Samples
ansible-playbook playbooks/install_benchmark_tools.yml -t cuda_samples

# 只安装 nvbandwidth
ansible-playbook playbooks/install_benchmark_tools.yml -t nvbandwidth

# 只安装 RDMA 工具
ansible-playbook playbooks/install_benchmark_tools.yml -t perftest
```

## 脚本的容错机制

我们的测试脚本已经考虑到这些工具可能不存在的情况：

```bash
# 示例：bandwidth_test.sh 中的检查逻辑
if command -v nvbandwidth &> /dev/null; then
    echo "Using nvbandwidth..."
    nvbandwidth -t host_to_device_memcpy_ce
else
    echo "WARNING: nvbandwidth not found, skipping this test"
fi
```

**特点**：
- ✅ 工具不存在时，会显示警告但不会中断执行
- ✅ 会尝试使用替代工具（如 nvbandwidth 不存在时使用 bandwidthTest）
- ✅ 会记录哪些测试被跳过

## 是否为专有/闭源工具？

### 答案：全部都是开源的 ✅

| 工具 | 开源状态 | 许可证 | 源码位置 |
|------|---------|--------|---------|
| nvbandwidth | 开源 | Apache 2.0 | https://github.com/NVIDIA/nvbandwidth |
| bandwidthTest | 开源 | BSD 3-Clause | https://github.com/NVIDIA/cuda-samples |
| p2pBandwidthLatencyTest | 开源 | BSD 3-Clause | https://github.com/NVIDIA/cuda-samples |
| perftest (ib_write_bw) | 开源 | GPL/BSD | https://github.com/linux-rdma/perftest |

**结论**：这些都是开源工具，但需要单独安装/编译。

## 建议

### 对于项目用户

1. **使用 Ansible 自动安装**（最简单）
   ```bash
   cd ansible
   ansible-playbook playbooks/install_benchmark_tools.yml
   ```

2. **验证安装**
   ```bash
   # 运行验证脚本
   ansible-playbook playbooks/validate_gpu.yml
   ```

3. **查看哪些工具已安装**
   ```bash
   ./scripts/validation/bandwidth_test.sh
   # 会显示哪些工具可用，哪些不可用
   ```

### 对于项目维护者

可以考虑：

1. **添加工具检查脚本**
   ```bash
   ./scripts/utils/check_dependencies.sh
   ```
   列出所有缺失的工具及安装方法

2. **更新文档**
   - 在 README 中明确说明需要先安装这些工具
   - 提供快速安装命令

3. **提供预编译二进制**
   - 对于常用平台（Ubuntu 20.04/22.04, Rocky 8/9）
   - 提供预编译的工具包

## 参考链接

- [NVIDIA nvbandwidth](https://github.com/NVIDIA/nvbandwidth)
- [CUDA Samples](https://github.com/NVIDIA/cuda-samples)
- [perftest](https://github.com/linux-rdma/perftest)
- [RDMA Core](https://github.com/linux-rdma/rdma-core)

## 总结

**问题的本质**：这些 bandwidth 测试命令都是**开源的专用工具**，但需要单独安装。它们不是 Linux 发行版的标准组件，也不随 CUDA Toolkit 默认安装。

**解决方案**：使用项目提供的 Ansible playbooks 一键安装所有工具。

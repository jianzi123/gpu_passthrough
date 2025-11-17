# 快速入门指南

5 分钟快速部署第一台 GPU 服务器，30 分钟完成小型集群部署。

## 🚀 5 分钟：单节点快速部署

### 前提条件

- Ubuntu 22.04 或 CentOS 8+
- 至少一块 NVIDIA GPU
- sudo 权限
- 网络连接

### 快速步骤

```bash
# 1. 克隆项目
git clone <repo-url>
cd gpu_passthrough

# 2. 安装 Ansible（如果没有）
sudo apt update && sudo apt install -y ansible  # Ubuntu
# sudo yum install -y ansible  # CentOS

# 3. 一键安装（自动检测 GPU 并安装最佳驱动）
cd ansible
ansible-playbook -i localhost, -c local playbooks/setup_gpu_baseline.yml

# 4. 重启（如果提示需要）
sudo reboot

# 5. 验证
nvidia-smi
```

**完成！** 你的 GPU 服务器已就绪。

---

## ⚡ 10 分钟：单节点 + 优化 + 验证

### 完整配置

```bash
# 1. 克隆并进入项目
git clone <repo-url>
cd gpu_passthrough

# 2. 完整优化部署
cd ansible
ansible-playbook -i localhost, -c local playbooks/full_deployment_optimized.yml

# 重启后继续...
sudo reboot

# 3. 运行完整验证
./scripts/validation/system_check.sh

# 4. 性能测试
sudo /usr/local/bin/gpu-benchmark bandwidth
sudo /usr/local/bin/gpu-benchmark nccl

# 5. 拉取 NGC 镜像（可选）
./scripts/utils/ngc_manager.sh pull pytorch
./scripts/utils/ngc_manager.sh test pytorch
```

### 验证清单

- ✅ nvidia-smi 运行正常
- ✅ CPU Governor 设置为 performance
- ✅ NUMA 配置正确
- ✅ PCIe/NVLink 带宽符合预期
- ✅ NGC 镜像可以访问 GPU

---

## 📦 30 分钟：小型集群部署（3-10 台）

### 步骤 1: 准备 Inventory（5 分钟）

```bash
# 创建主机清单
cd ansible/inventory
cat > hosts << EOF
[gpu_nodes]
gpu-01 ansible_host=192.168.1.101
gpu-02 ansible_host=192.168.1.102
gpu-03 ansible_host=192.168.1.103

[all:vars]
ansible_user=ubuntu
ansible_become=true
EOF
```

### 步骤 2: 配置 SSH 免密登录（5 分钟）

```bash
# 生成 SSH 密钥（如果没有）
ssh-keygen -t rsa -b 4096

# 复制公钥到所有节点
for i in gpu-0{1..3}; do
  ssh-copy-id ubuntu@${i}
done

# 测试连接
ansible -i inventory/hosts all -m ping
```

### 步骤 3: 批量部署（15 分钟）

```bash
# 方法 1: 标准部署
ansible-playbook -i inventory/hosts playbooks/setup_gpu_baseline.yml

# 方法 2: 完整优化部署（推荐）
ansible-playbook -i inventory/hosts playbooks/full_deployment_optimized.yml

# 如果需要重启
ansible -i inventory/hosts all -m reboot

# 等待重启完成
sleep 120
```

### 步骤 4: 验证集群（5 分钟）

```bash
# 检查所有节点的驱动
ansible -i inventory/hosts all -m shell -a "nvidia-smi --query-gpu=name,driver_version --format=csv"

# 运行验证 playbook
ansible-playbook -i inventory/hosts playbooks/validate_gpu.yml

# 检查报告
ls -lh /var/log/gpu_baseline/
```

### 完成！

你的 GPU 集群已经ready，可以开始训练了！

---

## 🎯 场景化快速入门

### 场景 1: 我只想快速测试一下

```bash
# 最小化安装
cd ansible
ansible-playbook -i localhost, -c local playbooks/setup_gpu_baseline.yml \
  -e "install_cuda=false" \
  -e "install_container_runtime=false" \
  -e "run_post_install_validation=false"

# 重启并验证
sudo reboot
nvidia-smi
```

### 场景 2: 我需要用于深度学习训练

```bash
# 完整部署 + NGC 镜像
ansible-playbook -i localhost, -c local playbooks/full_deployment_optimized.yml

# 重启
sudo reboot

# 拉取训练镜像
./scripts/utils/ngc_manager.sh pull pytorch
./scripts/utils/ngc_manager.sh pull nemo

# 测试训练
docker run --gpus all --rm nvcr.io/nvidia/pytorch:24.01-py3 \
  python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

### 场景 3: 我要部署大规模集群（50+ 台）

**推荐使用预编译驱动！**

```bash
# 1. 识别集群内核版本
ansible -i inventory/hosts all -m shell -a "uname -r" | \
  grep -v ">>" | sort -u > kernels.txt

# 2. 预构建驱动（一次性，2 小时）
./scripts/install/batch_build_drivers.sh

# 3. 建立驱动仓库
mkdir -p /var/www/drivers
cp /opt/precompiled-drivers/* /var/www/drivers/
cd /var/www/drivers && python3 -m http.server 8080 &

# 4. 分批部署（每批 10 台，每批 20 分钟）
ansible-playbook -i inventory/hosts playbooks/setup_gpu_baseline.yml \
  -e "driver_installation_method=precompiled" \
  -e "precompiled_repo=http://repo-server:8080" \
  --limit batch1 \
  --forks 10

# 重复 batch2, batch3...
```

**时间节省**: 传统方式 50 小时 → 预编译方式 3.5 小时（93% 节省！）

### 场景 4: Kubernetes GPU 节点

```bash
# 1. 部署驱动容器
ansible-playbook -i inventory/hosts playbooks/setup_gpu_baseline.yml \
  -e "driver_installation_method=driver-container"

# 2. 安装 GPU Operator
helm repo add nvidia https://nvidia.github.io/gpu-operator
helm install gpu-operator nvidia/gpu-operator \
  -n gpu-operator-resources \
  --create-namespace

# 3. 验证
kubectl get pods -n gpu-operator-resources
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
  restartPolicy: Never
EOF

kubectl logs gpu-test
```

---

## 🔧 常用命令速查

### 驱动管理

```bash
# 查看驱动版本
nvidia-smi --query-gpu=driver_version --format=csv,noheader

# 预编译驱动管理
./scripts/utils/manage_precompiled_drivers.sh list          # 列出可用驱动
./scripts/utils/manage_precompiled_drivers.sh install latest # 安装最新
./scripts/utils/manage_precompiled_drivers.sh rollback      # 回滚

# 驱动容器管理
systemctl status nvidia-driver    # 查看状态
systemctl restart nvidia-driver   # 重启
journalctl -u nvidia-driver -f    # 查看日志
```

### 验证和测试

```bash
# 快速验证
./scripts/validation/quick_check.sh

# 完整系统检查
./scripts/validation/system_check.sh

# 带宽测试
./scripts/validation/bandwidth_test.sh

# 性能基准
gpu-benchmark bandwidth   # PCIe/NVLink 带宽
gpu-benchmark nccl        # NCCL 通信
gpu-benchmark megatron    # 训练基准
```

### NGC 镜像管理

```bash
# 列出可用镜像
./scripts/utils/ngc_manager.sh list

# 拉取镜像
./scripts/utils/ngc_manager.sh pull pytorch
./scripts/utils/ngc_manager.sh pull nemo
./scripts/utils/ngc_manager.sh pull triton

# 运行镜像
./scripts/utils/ngc_manager.sh run pytorch

# 测试 GPU 访问
./scripts/utils/ngc_manager.sh test pytorch
```

### Ansible 快捷命令

```bash
# 检查所有节点 GPU
ansible -i inventory/hosts all -m shell -a "nvidia-smi -L"

# 检查驱动版本
ansible -i inventory/hosts all -m shell -a "nvidia-smi --query-gpu=driver_version --format=csv"

# 检查温度
ansible -i inventory/hosts all -m shell -a "nvidia-smi --query-gpu=temperature.gpu --format=csv"

# 运行脚本
ansible -i inventory/hosts all -m script -a "./scripts/validation/quick_check.sh"

# 重启所有节点
ansible -i inventory/hosts all -m reboot
```

---

## ⚠️ 常见问题

### Q: nvidia-smi 找不到

```bash
# 检查模块是否加载
lsmod | grep nvidia

# 如果没有，手动加载
sudo modprobe nvidia

# 如果仍然失败，检查安装
dpkg -l | grep nvidia-driver
```

### Q: 驱动安装后需要重启吗？

**是的**，首次安装驱动需要重启。可以在 playbook 中自动处理：

```bash
ansible-playbook -i inventory/hosts playbooks/setup_gpu_baseline.yml \
  -e "nvidia_driver_skip_reboot=false"
```

### Q: 如何选择驱动安装方法？

简单规则：
- **< 10 台**: Native（简单）
- **10-50 台**: Precompiled（推荐）
- **> 50 台**: Precompiled（必须）
- **Kubernetes**: Driver Container

### Q: 性能测试结果与基线差异大怎么办？

```bash
# 1. 检查 CPU 优化是否应用
./scripts/validation/system_check.sh | grep "CPU Governor"

# 2. 检查 NUMA 配置
nvidia-smi topo -m

# 3. 重新应用优化
ansible-playbook -i inventory/hosts playbooks/apply_cpu_optimization.yml

# 4. 重新测试
gpu-benchmark bandwidth
```

### Q: 怎么更新驱动？

```bash
# 预编译驱动
./scripts/utils/manage_precompiled_drivers.sh install <new-version>

# 如果出问题，回滚
./scripts/utils/manage_precompiled_drivers.sh rollback

# 驱动容器
sudo systemctl stop nvidia-driver
# 修改 /etc/systemd/system/nvidia-driver.service 中的版本
sudo systemctl daemon-reload
sudo systemctl start nvidia-driver
```

---

## 📚 下一步

完成快速入门后，建议阅读：

1. **详细文档**
   - [最佳实践指南](docs/best_practices.md) - 生产环境部署指南
   - [驱动安装方法](docs/gpu_driver_installation_methods.md) - 三种方法详解
   - [预编译驱动指南](docs/precompiled_driver_guide.md) - 大规模部署必读

2. **性能优化**
   - [CPU 优化](docs/latest_research_2025.md) - CPU 性能调优
   - [带宽测试](docs/bandwidth_and_benchmarks.md) - 带宽和基准测试

3. **NGC 容器**
   - [CUDA 兼容性和 NGC](docs/cuda_compatibility_and_ngc.md) - NGC 镜像使用

## 🆘 获取帮助

- 查看文档: `docs/`
- 运行示例: `examples/`
- 问题反馈: GitHub Issues
- 团队支持: gpu-team@company.com

---

**祝你使用愉快！GPU 集群已经准备就绪 🚀**

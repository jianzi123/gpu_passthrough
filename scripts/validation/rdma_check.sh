#!/bin/bash
# RDMA 环境全面验证脚本
# 检查 RDMA 驱动、设备、服务、GPUDirect RDMA 支持等
# 基于 InfiniBand/RoCE 最佳实践

set -euo pipefail

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 检查结果统计
PASS_COUNT=0
WARN_COUNT=0
FAIL_COUNT=0

# 输出文件
OUTPUT_DIR="${1:-/tmp/rdma_check_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTPUT_DIR"
OUTPUT_FILE="$OUTPUT_DIR/rdma_check.json"
SUMMARY_FILE="$OUTPUT_DIR/rdma_summary.md"
RESULTS=()

# 函数：添加检查结果
add_result() {
    local category="$1"
    local name="$2"
    local status="$3"
    local value="$4"
    local expected="$5"
    local details="${6:-}"

    # Escape quotes in JSON strings
    value=$(echo "$value" | sed 's/"/\\"/g')
    expected=$(echo "$expected" | sed 's/"/\\"/g')
    details=$(echo "$details" | sed 's/"/\\"/g')

    RESULTS+=("{\"category\":\"$category\",\"name\":\"$name\",\"status\":\"$status\",\"value\":\"$value\",\"expected\":\"$expected\",\"details\":\"$details\"}")

    case "$status" in
        "pass")
            echo -e "${GREEN}✓${NC} [$category] $name: $value"
            ((PASS_COUNT++))
            ;;
        "warn")
            echo -e "${YELLOW}⚠${NC} [$category] $name: $value (Expected: $expected)"
            ((WARN_COUNT++))
            ;;
        "fail")
            echo -e "${RED}✗${NC} [$category] $name: $value (Expected: $expected)"
            ((FAIL_COUNT++))
            ;;
        "info")
            echo -e "${CYAN}ℹ${NC} [$category] $name: $value"
            ;;
    esac

    # Add details if present
    if [ -n "$details" ]; then
        echo "    Details: $details"
    fi
}

# 打印分隔符
print_section() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_section "RDMA 环境验证"
echo "开始时间: $(date)"
echo "输出目录: $OUTPUT_DIR"
echo ""

#===========================================
# 1. RDMA 内核模块检查
#===========================================
print_section "1. RDMA 内核模块检查"

# 1.1 核心 RDMA 模块
echo -e "\n${BLUE}1.1 核心 RDMA 模块${NC}"

CORE_MODULES=("rdma_cm" "ib_core" "ib_uverbs" "rdma_ucm" "ib_umad")
for module in "${CORE_MODULES[@]}"; do
    if lsmod | grep -q "^$module"; then
        add_result "内核模块" "$module" "pass" "已加载" "已加载"
    else
        add_result "内核模块" "$module" "fail" "未加载" "已加载" "运行: modprobe $module"
    fi
done

# 1.2 InfiniBand 传输层模块
echo -e "\n${BLUE}1.2 InfiniBand 传输层模块${NC}"

TRANSPORT_MODULES=("ib_ipoib" "ib_srp" "ib_srpt" "ib_iser")
loaded_transport=0
for module in "${TRANSPORT_MODULES[@]}"; do
    if lsmod | grep -q "^$module"; then
        add_result "传输层模块" "$module" "pass" "已加载" "可选"
        ((loaded_transport++))
    fi
done

if [ $loaded_transport -eq 0 ]; then
    add_result "传输层模块" "任意传输模块" "warn" "未加载" "至少一个" "根据需要加载 IPoIB, SRP, iSER 等"
fi

# 1.3 厂商驱动模块
echo -e "\n${BLUE}1.3 厂商驱动模块${NC}"

VENDOR_MODULES=("mlx5_core" "mlx5_ib" "mlx4_core" "mlx4_ib")
loaded_vendor=0
for module in "${VENDOR_MODULES[@]}"; do
    if lsmod | grep -q "^$module"; then
        add_result "厂商驱动" "$module" "pass" "已加载" "已加载"
        ((loaded_vendor++))
    fi
done

if [ $loaded_vendor -eq 0 ]; then
    add_result "厂商驱动" "任意厂商驱动" "fail" "未加载" "至少一个" "未检测到 Mellanox/其他 RDMA 网卡驱动"
fi

# 1.4 GPUDirect RDMA 模块
echo -e "\n${BLUE}1.4 GPUDirect RDMA 支持${NC}"

if lsmod | grep -q "nv_peer_mem"; then
    nv_peer_version=$(modinfo nv_peer_mem 2>/dev/null | grep "^version:" | awk '{print $2}')
    add_result "GPUDirect" "nv_peer_mem" "pass" "已加载 (版本: ${nv_peer_version:-未知})" "已加载"
elif lsmod | grep -q "nvidia_peermem"; then
    add_result "GPUDirect" "nvidia_peermem" "pass" "已加载 (新版驱动)" "已加载"
else
    add_result "GPUDirect" "nv_peer_mem" "warn" "未加载" "已加载" "GPUDirect RDMA 不可用，需安装 nvidia-peer-memory 或 gdrcopy"
fi

#===========================================
# 2. RDMA 设备检查
#===========================================
print_section "2. RDMA 设备检查"

# 2.1 检查 ibstat 命令
echo -e "\n${BLUE}2.1 InfiniBand 设备状态${NC}"

if ! command -v ibstat &> /dev/null; then
    add_result "RDMA工具" "ibstat" "fail" "未安装" "已安装" "安装 infiniband-diags 包"
    IBSTAT_AVAILABLE=false
else
    add_result "RDMA工具" "ibstat" "pass" "已安装" "已安装"
    IBSTAT_AVAILABLE=true

    # 保存 ibstat 输出
    ibstat > "$OUTPUT_DIR/ibstat_output.txt" 2>&1 || true

    # 获取设备列表
    IB_DEVICES=$(ibstat -l 2>/dev/null || echo "")

    if [ -z "$IB_DEVICES" ]; then
        add_result "IB设备" "设备数量" "fail" "0" ">0" "未检测到 InfiniBand 设备"
    else
        device_count=$(echo "$IB_DEVICES" | wc -l)
        add_result "IB设备" "设备数量" "pass" "$device_count" ">0"

        # 检查每个设备的详细信息
        echo ""
        echo -e "${CYAN}检测到的 IB 设备:${NC}"
        while IFS= read -r device; do
            echo -e "\n  ${CYAN}设备: $device${NC}"

            # 获取端口数量
            port_count=$(ibstat "$device" 2>/dev/null | grep "Number of ports:" | awk '{print $4}')
            echo "    端口数量: ${port_count:-unknown}"

            # 检查每个端口
            for port in $(seq 1 ${port_count:-1}); do
                echo -e "    ${CYAN}端口 $port:${NC}"

                # 获取端口状态
                state=$(ibstat "$device" "$port" 2>/dev/null | grep "State:" | awk '{print $2}')
                rate=$(ibstat "$device" "$port" 2>/dev/null | grep "Rate:" | awk '{print $2, $3}')
                link_layer=$(ibstat "$device" "$port" 2>/dev/null | grep "Link layer:" | awk '{print $3}')
                physical_state=$(ibstat "$device" "$port" 2>/dev/null | grep "Physical state:" | awk '{$1=$2=""; print $0}' | xargs)

                echo "      状态: $state"
                echo "      速率: $rate"
                echo "      链路层: $link_layer"
                echo "      物理状态: $physical_state"

                if [ "$state" = "Active" ]; then
                    add_result "IB端口" "${device}:${port}" "pass" "Active @ $rate" "Active"
                elif [ "$state" = "Down" ]; then
                    add_result "IB端口" "${device}:${port}" "fail" "Down" "Active" "端口未连接或链路故障"
                else
                    add_result "IB端口" "${device}:${port}" "warn" "$state" "Active" "端口状态异常"
                fi
            done
        done <<< "$IB_DEVICES"
    fi
fi

# 2.2 检查 ibv_devinfo
echo -e "\n${BLUE}2.2 RDMA 设备详细信息${NC}"

if ! command -v ibv_devinfo &> /dev/null; then
    add_result "RDMA工具" "ibv_devinfo" "fail" "未安装" "已安装" "安装 libibverbs-dev 或 rdma-core-devel"
else
    add_result "RDMA工具" "ibv_devinfo" "pass" "已安装" "已安装"

    # 保存 ibv_devinfo 输出
    ibv_devinfo > "$OUTPUT_DIR/ibv_devinfo_output.txt" 2>&1 || true

    # 获取 RDMA 设备信息
    rdma_devices=$(ibv_devinfo -l 2>/dev/null | grep -v "hfi1" || echo "")

    if [ -z "$rdma_devices" ]; then
        add_result "RDMA设备" "libibverbs设备" "fail" "0" ">0" "未检测到 RDMA 设备"
    else
        device_count=$(echo "$rdma_devices" | wc -l)
        add_result "RDMA设备" "libibverbs设备" "pass" "$device_count" ">0"

        # 检查设备能力
        echo ""
        echo -e "${CYAN}RDMA 设备能力:${NC}"
        while IFS= read -r device; do
            echo -e "\n  ${CYAN}设备: $device${NC}"

            # 检查关键能力
            fw_version=$(ibv_devinfo -d "$device" 2>/dev/null | grep "fw_ver:" | awk '{print $2}')
            node_guid=$(ibv_devinfo -d "$device" 2>/dev/null | grep "node_guid:" | awk '{print $2}')
            max_qp=$(ibv_devinfo -d "$device" 2>/dev/null | grep "max_qp:" | awk '{print $2}')
            max_cq=$(ibv_devinfo -d "$device" 2>/dev/null | grep "max_cq:" | awk '{print $2}')

            echo "    固件版本: ${fw_version:-unknown}"
            echo "    Node GUID: ${node_guid:-unknown}"
            echo "    最大 QP: ${max_qp:-unknown}"
            echo "    最大 CQ: ${max_cq:-unknown}"

            add_result "设备能力" "${device}_firmware" "info" "$fw_version" "-"
        done <<< "$rdma_devices"
    fi
fi

# 2.3 使用 rdma link 检查 (如果可用)
echo -e "\n${BLUE}2.3 RDMA 链路状态 (rdma tool)${NC}"

if command -v rdma &> /dev/null; then
    add_result "RDMA工具" "rdma" "pass" "已安装" "已安装"

    rdma link > "$OUTPUT_DIR/rdma_link_output.txt" 2>&1 || true

    # 显示链路状态
    if rdma link show &> /dev/null; then
        echo ""
        rdma link show
    fi
else
    add_result "RDMA工具" "rdma" "warn" "未安装" "已安装" "安装 iproute2 包获取 rdma 工具"
fi

#===========================================
# 3. RDMA 服务和软件栈检查
#===========================================
print_section "3. RDMA 软件栈检查"

# 3.1 检查关键包
echo -e "\n${BLUE}3.1 关键软件包${NC}"

REQUIRED_PACKAGES=("libibverbs" "librdmacm" "rdma-core")
OPTIONAL_PACKAGES=("infiniband-diags" "perftest" "opensm")

for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if dpkg -l | grep -q "^ii.*$pkg" 2>/dev/null || rpm -q "$pkg" &>/dev/null; then
        pkg_version=$(dpkg -l | grep "^ii.*$pkg" | awk '{print $3}' 2>/dev/null || rpm -q "$pkg" 2>/dev/null | head -1)
        add_result "软件包" "$pkg" "pass" "已安装 ($pkg_version)" "已安装"
    else
        add_result "软件包" "$pkg" "fail" "未安装" "已安装" "安装 $pkg"
    fi
done

echo -e "\n${BLUE}3.2 可选软件包${NC}"
for pkg in "${OPTIONAL_PACKAGES[@]}"; do
    if dpkg -l | grep -q "^ii.*$pkg" 2>/dev/null || rpm -q "$pkg" &>/dev/null; then
        add_result "可选软件包" "$pkg" "pass" "已安装" "可选"
    else
        add_result "可选软件包" "$pkg" "warn" "未安装" "推荐安装" "$pkg 用于诊断和性能测试"
    fi
done

# 3.2 检查 perftest 工具
echo -e "\n${BLUE}3.3 性能测试工具${NC}"

PERFTEST_TOOLS=("ib_write_bw" "ib_read_bw" "ib_send_bw" "ib_write_lat")
perftest_found=0
for tool in "${PERFTEST_TOOLS[@]}"; do
    if command -v "$tool" &> /dev/null; then
        add_result "性能工具" "$tool" "pass" "可用" "可用"
        ((perftest_found++))
    fi
done

if [ $perftest_found -eq 0 ]; then
    add_result "性能工具" "perftest" "warn" "未安装" "推荐安装" "安装 perftest 包用于 RDMA 带宽测试"
fi

# 3.3 Subnet Manager 检查
echo -e "\n${BLUE}3.4 Subnet Manager${NC}"

if command -v opensm &> /dev/null; then
    add_result "子网管理" "opensm" "pass" "已安装" "已安装"

    # 检查 opensm 是否运行
    if pgrep -x opensm > /dev/null; then
        add_result "子网管理" "opensm服务" "pass" "运行中" "运行中"
    else
        add_result "子网管理" "opensm服务" "warn" "未运行" "运行中" "InfiniBand 需要 Subnet Manager，可能在交换机上运行"
    fi
else
    add_result "子网管理" "opensm" "info" "未安装" "可选" "小型网络可在一台主机运行 opensm"
fi

#===========================================
# 4. 网络配置检查
#===========================================
print_section "4. 网络配置检查"

# 4.1 IPoIB 接口
echo -e "\n${BLUE}4.1 IPoIB 网络接口${NC}"

ipoib_interfaces=$(ip link show | grep -o 'ib[0-9]*' | sort -u || echo "")

if [ -z "$ipoib_interfaces" ]; then
    add_result "网络接口" "IPoIB" "warn" "未配置" "可选" "可配置 IPoIB 用于 IP 网络"
else
    ipoib_count=$(echo "$ipoib_interfaces" | wc -l)
    add_result "网络接口" "IPoIB接口数" "pass" "$ipoib_count" ">0"

    echo ""
    echo -e "${CYAN}IPoIB 接口详情:${NC}"
    while IFS= read -r iface; do
        if [ -n "$iface" ]; then
            echo -e "\n  ${CYAN}接口: $iface${NC}"

            # 获取接口状态
            state=$(ip link show "$iface" | grep -o "state [A-Z]*" | awk '{print $2}')
            mtu=$(ip link show "$iface" | grep -o "mtu [0-9]*" | awk '{print $2}')
            ip_addr=$(ip addr show "$iface" | grep "inet " | awk '{print $2}' | head -1)

            echo "    状态: ${state:-unknown}"
            echo "    MTU: ${mtu:-unknown}"
            echo "    IP: ${ip_addr:-未配置}"

            if [ "$state" = "UP" ]; then
                add_result "IPoIB状态" "$iface" "pass" "UP (MTU: $mtu)" "UP"

                # 检查 MTU (推荐 65520 for connected mode)
                if [ "${mtu:-0}" -ge 65520 ]; then
                    add_result "IPoIB_MTU" "$iface" "pass" "$mtu" ">=65520"
                elif [ "${mtu:-0}" -ge 2044 ]; then
                    add_result "IPoIB_MTU" "$iface" "warn" "$mtu" ">=65520" "推荐使用 connected mode (MTU 65520)"
                else
                    add_result "IPoIB_MTU" "$iface" "warn" "$mtu" ">=2044" "MTU 配置偏低"
                fi
            else
                add_result "IPoIB状态" "$iface" "warn" "$state" "UP" "接口未启用"
            fi
        fi
    done <<< "$ipoib_interfaces"
fi

# 4.2 RoCE 接口检查
echo -e "\n${BLUE}4.2 RoCE (RDMA over Ethernet) 接口${NC}"

if command -v rdma &> /dev/null; then
    roce_links=$(rdma link show 2>/dev/null | grep -i "roce" || echo "")

    if [ -n "$roce_links" ]; then
        add_result "RoCE" "RoCE接口" "pass" "检测到" "可选"
        echo ""
        echo -e "${CYAN}RoCE 接口:${NC}"
        echo "$roce_links"
    else
        add_result "RoCE" "RoCE接口" "info" "未检测到" "可选"
    fi
fi

#===========================================
# 5. GPUDirect RDMA 环境检查
#===========================================
print_section "5. GPUDirect RDMA 环境"

# 5.1 检查 GPU 是否存在
echo -e "\n${BLUE}5.1 GPU 检测${NC}"

if command -v nvidia-smi &> /dev/null; then
    gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
    add_result "GPU" "GPU数量" "pass" "$gpu_count" ">0"

    # 获取 GPU 型号
    gpu_model=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo "  GPU 型号: $gpu_model"
else
    add_result "GPU" "nvidia-smi" "warn" "未安装" "可选" "GPUDirect RDMA 需要 NVIDIA GPU"
fi

# 5.2 检查 NVIDIA 驱动和 GPUDirect RDMA 支持
echo -e "\n${BLUE}5.2 NVIDIA 驱动和 GPUDirect${NC}"

if command -v nvidia-smi &> /dev/null; then
    driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    add_result "GPU驱动" "NVIDIA驱动版本" "pass" "$driver_version" ">450.x"

    # 检查 nv_peer_mem 或 nvidia_peermem
    if lsmod | grep -q "nv_peer_mem\|nvidia_peermem"; then
        # 检查 /sys/kernel/mm/memory_peer_target
        if [ -d /sys/kernel/mm/memory_peer_target ]; then
            peer_targets=$(ls /sys/kernel/mm/memory_peer_target/ 2>/dev/null | wc -l)
            add_result "GPUDirect" "peer_memory目标" "pass" "$peer_targets" ">0"
        fi

        # 检查 GPU 内存是否可以被 RDMA 访问
        if command -v nvidia-smi &> /dev/null && command -v ibv_devinfo &> /dev/null; then
            add_result "GPUDirect" "环境检查" "pass" "nv_peer_mem已加载且GPU可用" "完整"
        fi
    else
        add_result "GPUDirect" "peer_memory模块" "warn" "未加载" "已加载" "需要 nvidia-peer-memory 包"
    fi
fi

# 5.3 检查 GPU 和 RDMA 设备的 NUMA 亲和性
echo -e "\n${BLUE}5.3 NUMA 亲和性${NC}"

if command -v nvidia-smi &> /dev/null && [ "$IBSTAT_AVAILABLE" = true ]; then
    echo ""
    echo -e "${CYAN}GPU NUMA 节点:${NC}"
    nvidia-smi topo -m > "$OUTPUT_DIR/gpu_topology.txt" 2>&1 || true

    # 获取每个 GPU 的 NUMA 节点
    for gpu_id in $(seq 0 $((gpu_count - 1))); do
        numa_node=$(nvidia-smi -i "$gpu_id" --query-gpu=numa_node --format=csv,noheader 2>/dev/null || echo "N/A")
        echo "  GPU $gpu_id: NUMA node $numa_node"
    done

    # 获取 IB 设备的 NUMA 节点 (如果可用)
    if [ -n "${IB_DEVICES:-}" ]; then
        echo ""
        echo -e "${CYAN}IB 设备 NUMA 节点:${NC}"
        while IFS= read -r device; do
            # 从 sysfs 获取 NUMA 节点
            ib_pci=$(ibv_devinfo -d "$device" 2>/dev/null | grep "node_guid" | head -1 || echo "")

            # 尝试从 /sys 找到设备
            for pci_dev in /sys/class/infiniband/"$device"/device; do
                if [ -e "$pci_dev/numa_node" ]; then
                    numa_node=$(cat "$pci_dev/numa_node" 2>/dev/null || echo "N/A")
                    echo "  $device: NUMA node $numa_node"

                    add_result "NUMA亲和性" "$device" "info" "NUMA $numa_node" "-"
                fi
            done
        done <<< "$IB_DEVICES"
    fi
fi

#===========================================
# 6. 系统配置和性能参数
#===========================================
print_section "6. 系统配置和性能参数"

# 6.1 IOMMU 配置
echo -e "\n${BLUE}6.1 IOMMU 配置${NC}"

if [ -d /sys/class/iommu ]; then
    iommu_enabled=true
    iommu_count=$(ls /sys/class/iommu/ | wc -l)
    add_result "IOMMU" "IOMMU状态" "pass" "已启用 ($iommu_count groups)" "已启用"
else
    add_result "IOMMU" "IOMMU状态" "warn" "未启用" "已启用" "RDMA 性能可能受影响，检查 BIOS 和内核参数"
fi

# 检查内核启动参数
if grep -q "iommu=pt\|intel_iommu=on\|amd_iommu=on" /proc/cmdline 2>/dev/null; then
    cmdline_iommu=$(grep -o "iommu=[^ ]*\|intel_iommu=[^ ]*\|amd_iommu=[^ ]*" /proc/cmdline | tr '\n' ' ')
    add_result "IOMMU" "内核参数" "pass" "$cmdline_iommu" "已配置"
else
    add_result "IOMMU" "内核参数" "warn" "未配置" "已配置" "添加 intel_iommu=on 或 amd_iommu=on 到内核参数"
fi

# 6.2 内存锁定限制
echo -e "\n${BLUE}6.2 内存锁定限制${NC}"

memlock_hard=$(ulimit -Hl 2>/dev/null || echo "unknown")
memlock_soft=$(ulimit -Sl 2>/dev/null || echo "unknown")

if [ "$memlock_hard" = "unlimited" ] || [ "$memlock_hard" -gt 1000000 ] 2>/dev/null; then
    add_result "内存锁定" "hard_limit" "pass" "$memlock_hard" "unlimited或足够大"
else
    add_result "内存锁定" "hard_limit" "fail" "$memlock_hard" "unlimited" "需要在 /etc/security/limits.conf 设置"
fi

if [ "$memlock_soft" = "unlimited" ] || [ "$memlock_soft" -gt 1000000 ] 2>/dev/null; then
    add_result "内存锁定" "soft_limit" "pass" "$memlock_soft" "unlimited或足够大"
else
    add_result "内存锁定" "soft_limit" "warn" "$memlock_soft" "unlimited" "建议设置为 unlimited"
fi

# 检查 limits.conf
if grep -q "memlock.*unlimited" /etc/security/limits.conf 2>/dev/null; then
    add_result "内存锁定" "limits.conf" "pass" "已配置" "已配置"
else
    add_result "内存锁定" "limits.conf" "warn" "未配置" "已配置" "添加 '* soft memlock unlimited' 和 '* hard memlock unlimited'"
fi

# 6.3 PCIe 配置
echo -e "\n${BLUE}6.3 PCIe 配置${NC}"

if command -v nvidia-smi &> /dev/null; then
    echo -e "\n${CYAN}GPU PCIe 状态:${NC}"
    nvidia-smi --query-gpu=index,pci.bus_id,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max --format=csv
fi

if [ -n "${IB_DEVICES:-}" ] && command -v lspci &> /dev/null; then
    echo -e "\n${CYAN}IB 设备 PCIe 状态:${NC}"
    while IFS= read -r device; do
        # 从 sysfs 获取 PCI 地址
        for pci_path in /sys/class/infiniband/"$device"/device; do
            if [ -e "$pci_path" ]; then
                pci_addr=$(basename "$(readlink -f "$pci_path")")

                # 使用 lspci 获取详细信息
                lspci_info=$(lspci -s "$pci_addr" -vv 2>/dev/null | grep "LnkSta:" | head -1)
                if [ -n "$lspci_info" ]; then
                    echo "  $device ($pci_addr): $lspci_info"
                fi
            fi
        done
    done <<< "$IB_DEVICES"
fi

#===========================================
# 7. 生成 JSON 报告
#===========================================
echo ""
echo "生成 JSON 报告..."

# 构建 JSON
JSON_OUTPUT='{'
JSON_OUTPUT+='"timestamp":"'$(date -u +%Y-%m-%dT%H:%M:%SZ)'",'
JSON_OUTPUT+='"hostname":"'$(hostname)'",'
JSON_OUTPUT+='"checks":['

# 添加所有结果
first=true
for result in "${RESULTS[@]}"; do
    if [ "$first" = true ]; then
        first=false
    else
        JSON_OUTPUT+=','
    fi
    JSON_OUTPUT+="$result"
done

JSON_OUTPUT+='],'
JSON_OUTPUT+='"summary":{'
JSON_OUTPUT+='"total":'$((PASS_COUNT + WARN_COUNT + FAIL_COUNT))','
JSON_OUTPUT+='"passed":'$PASS_COUNT','
JSON_OUTPUT+='"warnings":'$WARN_COUNT','
JSON_OUTPUT+='"failed":'$FAIL_COUNT
JSON_OUTPUT+='}'
JSON_OUTPUT+='}'

# 写入 JSON 文件
echo "$JSON_OUTPUT" | python3 -m json.tool > "$OUTPUT_FILE" 2>/dev/null || echo "$JSON_OUTPUT" > "$OUTPUT_FILE"

#===========================================
# 8. 生成 Markdown 摘要报告
#===========================================
echo "生成 Markdown 摘要报告..."

cat > "$SUMMARY_FILE" << EOF
# RDMA 环境验证摘要报告

**生成时间**: $(date)
**主机名**: $(hostname)
**输出目录**: $OUTPUT_DIR

## 总体状态

- ✅ **通过**: $PASS_COUNT 项
- ⚠️  **警告**: $WARN_COUNT 项
- ❌ **失败**: $FAIL_COUNT 项

## RDMA 环境就绪状态

EOF

# 判断 RDMA 环境是否就绪
if [ $FAIL_COUNT -eq 0 ] && [ $WARN_COUNT -le 5 ]; then
    echo "### ✅ RDMA 环境基本就绪" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
    echo "您的 RDMA 环境已基本配置完成，可以进行基本的 RDMA 通信测试。" >> "$SUMMARY_FILE"
elif [ $FAIL_COUNT -le 3 ]; then
    echo "### ⚠️ RDMA 环境部分就绪" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
    echo "您的 RDMA 环境存在一些问题，建议修复后再进行生产使用。" >> "$SUMMARY_FILE"
else
    echo "### ❌ RDMA 环境未就绪" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
    echo "您的 RDMA 环境存在严重问题，需要修复后才能使用。" >> "$SUMMARY_FILE"
fi

echo "" >> "$SUMMARY_FILE"
echo "## 关键检查项" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# 添加关键检查项到报告
cat >> "$SUMMARY_FILE" << 'EOF'
### 内核模块

- RDMA 核心模块 (rdma_cm, ib_core, ib_uverbs)
- 厂商驱动模块 (mlx5_core, mlx4_core)
- GPUDirect RDMA 模块 (nv_peer_mem)

### RDMA 设备

- InfiniBand 设备状态
- 端口状态和链路速度
- RDMA 设备能力

### 软件栈

- libibverbs, librdmacm, rdma-core
- 性能测试工具 (perftest)
- 诊断工具 (infiniband-diags)

### 网络配置

- IPoIB 接口配置
- MTU 设置
- RoCE 支持 (如适用)

### GPUDirect RDMA

- NVIDIA GPU 和驱动
- nv_peer_mem 模块
- GPU 和 IB 设备 NUMA 亲和性

### 系统配置

- IOMMU 启用和配置
- 内存锁定限制
- PCIe 配置

## 建议的修复措施

EOF

# 添加具体的修复建议
if [ $FAIL_COUNT -gt 0 ] || [ $WARN_COUNT -gt 0 ]; then
    echo "基于检测结果，以下是建议的修复措施：" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"

    # 从检查结果中提取失败和警告项
    echo "$JSON_OUTPUT" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    fails = [c for c in data['checks'] if c['status'] == 'fail']
    warns = [c for c in data['checks'] if c['status'] == 'warn']

    if fails:
        print('### 必须修复的问题\n')
        for i, item in enumerate(fails, 1):
            print(f\"{i}. **[{item['category']}] {item['name']}**\")
            print(f\"   - 当前值: {item['value']}\")
            print(f\"   - 期望值: {item['expected']}\")
            if item.get('details'):
                print(f\"   - 建议: {item['details']}\")
            print()

    if warns:
        print('### 建议修复的问题\n')
        for i, item in enumerate(warns, 1):
            print(f\"{i}. **[{item['category']}] {item['name']}**\")
            print(f\"   - 当前值: {item['value']}\")
            print(f\"   - 期望值: {item['expected']}\")
            if item.get('details'):
                print(f\"   - 建议: {item['details']}\")
            print()
except:
    pass
" >> "$SUMMARY_FILE" 2>/dev/null || echo "无法生成详细建议" >> "$SUMMARY_FILE"
else
    echo "✅ 未发现需要修复的问题。" >> "$SUMMARY_FILE"
fi

cat >> "$SUMMARY_FILE" << EOF

## 常用 RDMA 测试命令

### 1. 测试 RDMA 带宽 (需要两台主机)

**服务端 (节点1)**:
\`\`\`bash
ib_write_bw
\`\`\`

**客户端 (节点2)**:
\`\`\`bash
ib_write_bw <server_ip>
\`\`\`

### 2. 测试 GPUDirect RDMA 带宽

**服务端**:
\`\`\`bash
ib_write_bw --use_cuda=0
\`\`\`

**客户端**:
\`\`\`bash
ib_write_bw --use_cuda=0 <server_ip>
\`\`\`

### 3. 测试延迟

\`\`\`bash
# 服务端
ib_write_lat

# 客户端
ib_write_lat <server_ip>
\`\`\`

### 4. 检查 IB 端口状态

\`\`\`bash
ibstat
ibv_devinfo
rdma link show
\`\`\`

### 5. 检查 GPUDirect RDMA

\`\`\`bash
# 检查 nv_peer_mem 模块
lsmod | grep nv_peer_mem

# 检查 peer memory targets
ls /sys/kernel/mm/memory_peer_target/
\`\`\`

## 输出文件

- **JSON 报告**: $OUTPUT_FILE
- **Markdown 摘要**: $SUMMARY_FILE
- **ibstat 输出**: $OUTPUT_DIR/ibstat_output.txt
- **ibv_devinfo 输出**: $OUTPUT_DIR/ibv_devinfo_output.txt
- **GPU 拓扑**: $OUTPUT_DIR/gpu_topology.txt

## 参考资源

- [NVIDIA GPUDirect RDMA 文档](https://docs.nvidia.com/cuda/gpudirect-rdma/)
- [Mellanox OFED 文档](https://docs.nvidia.com/networking/display/mlnxofedv24010331)
- [RDMA Core 用户指南](https://github.com/linux-rdma/rdma-core)
- [InfiniBand 性能调优指南](https://docs.nvidia.com/networking/display/perftuning)

---
**报告生成于**: $(date)
EOF

#===========================================
# 9. 最终摘要
#===========================================
print_section "验证完成"

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}检查结果摘要${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "  ${GREEN}✓ 通过${NC}:  $PASS_COUNT"
echo -e "  ${YELLOW}⚠ 警告${NC}:  $WARN_COUNT"
echo -e "  ${RED}✗ 失败${NC}:  $FAIL_COUNT"
echo ""

# 判断并显示总体状态
if [ $FAIL_COUNT -eq 0 ] && [ $WARN_COUNT -le 5 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✅ RDMA 环境基本就绪${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "您的系统已经具备基本的 RDMA 功能，可以开始测试。"
elif [ $FAIL_COUNT -le 3 ]; then
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}⚠️ RDMA 环境部分就绪${NC}"
    echo -e "${YELLOW}========================================${NC}"
    echo ""
    echo "建议修复上述警告和失败项后再进行生产使用。"
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}❌ RDMA 环境未就绪${NC}"
    echo -e "${RED}========================================${NC}"
    echo ""
    echo "请先解决上述失败项后再使用 RDMA 功能。"
fi

echo ""
echo "详细报告已保存至:"
echo "  📄 JSON 报告: $OUTPUT_FILE"
echo "  📋 Markdown 摘要: $SUMMARY_FILE"
echo "  📁 输出目录: $OUTPUT_DIR"
echo ""
echo "查看 Markdown 摘要:"
echo "  cat $SUMMARY_FILE"
echo ""
echo "查看 JSON 报告:"
echo "  cat $OUTPUT_FILE | python3 -m json.tool"
echo ""

exit 0

#!/bin/bash
# 全面的 GPU 服务器系统验证脚本
# 包含 CPU、NUMA、IOMMU、PCIe、GPU 等所有配置检查
# 基于 2024-2025 最佳实践

set -euo pipefail

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 检查结果统计
PASS_COUNT=0
WARN_COUNT=0
FAIL_COUNT=0

# 输出文件
OUTPUT_FILE="${1:-/tmp/system_check_$(date +%Y%m%d_%H%M%S).json}"
RESULTS=()

# 函数：添加检查结果
add_result() {
    local category="$1"
    local name="$2"
    local status="$3"
    local value="$4"
    local expected="$5"
    local details="${6:-}"

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
    esac
}

# 打印分隔符
print_section() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_section "GPU Server System Validation"
echo "Started at: $(date)"
echo "Output: $OUTPUT_FILE"

#===========================================
# 1. CPU 配置检查
#===========================================
print_section "1. CPU Configuration"

# 1.1 CPU Governor
echo -e "\n${BLUE}1.1 CPU Governor${NC}"
governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "unknown")
if [ "$governor" = "performance" ]; then
    add_result "CPU" "CPU_Governor" "pass" "$governor" "performance"
else
    add_result "CPU" "CPU_Governor" "warn" "$governor" "performance" "Set to performance for best GPU workload performance"
fi

# 检查所有 CPU 的 governor
governors_count=$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | sort | uniq -c || echo "")
if [ -n "$governors_count" ]; then
    echo "  Governor distribution:"
    echo "$governors_count" | while read count gov; do
        echo "    $gov: $count cores"
    done
fi

# 1.2 Turbo Boost / Turbo Core
echo -e "\n${BLUE}1.2 Turbo Boost Status${NC}"
if [ -f /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
    turbo=$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)
    if [ "$turbo" -eq 0 ]; then
        add_result "CPU" "Intel_Turbo_Boost" "pass" "enabled" "enabled"
    else
        add_result "CPU" "Intel_Turbo_Boost" "fail" "disabled" "enabled"
    fi
elif [ -f /sys/devices/system/cpu/cpufreq/boost ]; then
    boost=$(cat /sys/devices/system/cpu/cpufreq/boost)
    if [ "$boost" -eq 1 ]; then
        add_result "CPU" "AMD_Turbo_Core" "pass" "enabled" "enabled"
    else
        add_result "CPU" "AMD_Turbo_Core" "fail" "disabled" "enabled"
    fi
else
    add_result "CPU" "Turbo_Technology" "warn" "unknown" "enabled" "Cannot detect turbo status"
fi

# 1.3 CPU 频率
echo -e "\n${BLUE}1.3 CPU Frequency${NC}"
if command -v cpupower &> /dev/null; then
    current_freq=$(cpupower frequency-info | grep "current CPU frequency" | awk '{print $4, $5}' | head -1)
    max_freq=$(cpupower frequency-info | grep "hardware limits" | awk '{print $5, $6}' | head -1)
    echo "  Current: $current_freq"
    echo "  Max: $max_freq"
    add_result "CPU" "CPU_Frequency" "pass" "$current_freq" "close_to_max"
else
    freq=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null || echo "0")
    freq_mhz=$((freq / 1000))
    echo "  Current: ${freq_mhz} MHz"
    add_result "CPU" "CPU_Frequency" "pass" "${freq_mhz} MHz" "high"
fi

# 1.4 C-States
echo -e "\n${BLUE}1.4 C-States${NC}"
if command -v cpupower &> /dev/null; then
    idle_states=$(cpupower idle-info 2>/dev/null | grep "Number of idle states" | awk '{print $5}' || echo "unknown")
    echo "  Idle states: $idle_states"
    if [ "$idle_states" -le 2 ] 2>/dev/null; then
        add_result "CPU" "C-States" "pass" "$idle_states states" "<=2 (C0/C1 only)"
    elif [ "$idle_states" = "unknown" ]; then
        add_result "CPU" "C-States" "warn" "unknown" "<=2" "Cannot determine C-States"
    else
        add_result "CPU" "C-States" "warn" "$idle_states states" "<=2" "Consider disabling deep C-States for lower latency"
    fi
else
    add_result "CPU" "C-States" "warn" "unknown" "<=2" "cpupower not installed"
fi

#===========================================
# 2. NUMA 配置检查
#===========================================
print_section "2. NUMA Configuration"

# 2.1 NUMA 节点数
echo -e "\n${BLUE}2.1 NUMA Topology${NC}"
if command -v numactl &> /dev/null; then
    numa_nodes=$(numactl --hardware | grep "available:" | awk '{print $2}' || echo "0")
    echo "  NUMA Nodes: $numa_nodes"

    if [ "$numa_nodes" -gt 0 ]; then
        add_result "NUMA" "NUMA_Nodes" "pass" "$numa_nodes" ">0"

        # 显示 NUMA 拓扑
        echo "  NUMA distances:"
        numactl --hardware | grep "node distances" -A $((numa_nodes + 1))
    else
        add_result "NUMA" "NUMA_Nodes" "warn" "0" ">0" "NUMA not configured or not supported"
    fi
else
    add_result "NUMA" "NUMA_Tools" "fail" "numactl not installed" "installed" "Install numactl package"
fi

# 2.2 GPU NUMA 亲和性
echo -e "\n${BLUE}2.2 GPU NUMA Affinity${NC}"
if command -v nvidia-smi &> /dev/null; then
    # 检查每个 GPU 的 NUMA 节点
    gpu_count=$(nvidia-smi -L | wc -l)
    echo "  GPUs detected: $gpu_count"

    for i in $(seq 0 $((gpu_count - 1))); do
        pci_id=$(nvidia-smi --id=$i --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null)

        if [ -n "$pci_id" ]; then
            # 转换 PCI ID 格式
            pci_addr=$(echo $pci_id | tr '[:upper:]' '[:lower:]' | sed 's/^0000://')
            numa_node=$(cat /sys/bus/pci/devices/0000:$pci_addr/numa_node 2>/dev/null || echo "-1")

            echo "  GPU $i (PCI: $pci_id): NUMA Node $numa_node"

            if [ "$numa_node" -ge 0 ] 2>/dev/null; then
                add_result "NUMA" "GPU_${i}_NUMA" "pass" "node $numa_node" ">=0"
            else
                add_result "NUMA" "GPU_${i}_NUMA" "warn" "not assigned" "assigned" "GPU not assigned to NUMA node"
            fi
        fi
    done

    # 显示 GPU 拓扑
    echo -e "\n  GPU Topology:"
    nvidia-smi topo -m 2>/dev/null || echo "  (topology matrix not available)"
else
    add_result "NUMA" "GPU_Detection" "warn" "nvidia-smi not available" "available"
fi

#===========================================
# 3. IOMMU 配置检查
#===========================================
print_section "3. IOMMU Configuration"

# 3.1 IOMMU 启用状态
echo -e "\n${BLUE}3.1 IOMMU Status${NC}"
if dmesg | grep -q "IOMMU enabled\|DMAR: Intel"; then
    iommu_status="enabled"
    add_result "IOMMU" "IOMMU_Enabled" "pass" "enabled" "enabled"

    # 显示 IOMMU 信息
    echo "  IOMMU messages:"
    dmesg | grep -i "IOMMU\|DMAR" | tail -5 | sed 's/^/    /'
elif dmesg | grep -q "AMD-Vi"; then
    iommu_status="enabled (AMD-Vi)"
    add_result "IOMMU" "IOMMU_Enabled" "pass" "enabled (AMD-Vi)" "enabled"

    echo "  AMD-Vi messages:"
    dmesg | grep -i "AMD-Vi" | tail -5 | sed 's/^/    /'
else
    iommu_status="disabled or not found"
    add_result "IOMMU" "IOMMU_Enabled" "fail" "disabled" "enabled" "Enable Intel VT-d or AMD IOMMU in BIOS"
fi

# 3.2 IOMMU 组
echo -e "\n${BLUE}3.2 IOMMU Groups${NC}"
if [ -d /sys/kernel/iommu_groups ]; then
    iommu_group_count=$(find /sys/kernel/iommu_groups/ -maxdepth 1 -type d | wc -l)
    echo "  IOMMU groups: $((iommu_group_count - 1))"

    # 显示 GPU 的 IOMMU 组
    echo "  GPU IOMMU Groups:"
    for d in /sys/kernel/iommu_groups/*/devices/*; do
        if [[ -e $d ]]; then
            device=$(basename $d)
            if lspci -nns $device 2>/dev/null | grep -q "NVIDIA\|AMD.*Radeon"; then
                group=$(echo $d | awk -F'/' '{print $(NF-2)}')
                device_info=$(lspci -nns $device 2>/dev/null | head -1)
                echo "    Group $group: $device_info"
            fi
        fi
    done

    add_result "IOMMU" "IOMMU_Groups" "pass" "$((iommu_group_count - 1))" ">0"
else
    add_result "IOMMU" "IOMMU_Groups" "warn" "not found" "present"
fi

# 3.3 内核参数检查
echo -e "\n${BLUE}3.3 Kernel Parameters${NC}"
cmdline=$(cat /proc/cmdline)

# 检查 IOMMU 参数
if echo "$cmdline" | grep -q "intel_iommu=on\|amd_iommu=on"; then
    add_result "IOMMU" "Kernel_IOMMU_Param" "pass" "present" "present"
else
    add_result "IOMMU" "Kernel_IOMMU_Param" "warn" "missing" "intel_iommu=on or amd_iommu=on"
fi

# 检查 IOMMU passthrough
if echo "$cmdline" | grep -q "iommu=pt"; then
    add_result "IOMMU" "IOMMU_Passthrough" "pass" "enabled" "enabled"
else
    add_result "IOMMU" "IOMMU_Passthrough" "warn" "disabled" "enabled" "Add iommu=pt for better performance"
fi

#===========================================
# 4. PCIe 配置检查
#===========================================
print_section "4. PCIe Configuration"

# 4.1 GPU PCIe 链路状态
echo -e "\n${BLUE}4.1 GPU PCIe Link Status${NC}"
if command -v nvidia-smi &> /dev/null; then
    echo "  GPU PCIe Link Information:"
    nvidia-smi --query-gpu=index,name,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max --format=csv | while IFS=, read -r idx name gen_curr gen_max width_curr width_max; do
        if [ "$idx" != "index" ]; then
            echo "    GPU $idx: Gen$gen_curr/$gen_max x$width_curr/$width_max"

            # 检查是否运行在最大速度
            gen_curr_trim=$(echo $gen_curr | tr -d ' ')
            gen_max_trim=$(echo $gen_max | tr -d ' ')
            width_curr_trim=$(echo $width_curr | tr -d ' ')
            width_max_trim=$(echo $width_max | tr -d ' ')

            if [ "$gen_curr_trim" = "$gen_max_trim" ] && [ "$width_curr_trim" = "$width_max_trim" ]; then
                add_result "PCIe" "GPU_${idx}_PCIe_Link" "pass" "Gen$gen_curr x$width_curr" "max"
            else
                add_result "PCIe" "GPU_${idx}_PCIe_Link" "warn" "Gen$gen_curr x$width_curr" "Gen$gen_max x$width_max" "Not running at maximum PCIe speed"
            fi
        fi
    done
fi

# 4.2 PCIe 错误检查
echo -e "\n${BLUE}4.2 PCIe Errors${NC}"
if command -v nvidia-smi &> /dev/null; then
    pcie_errors=$(nvidia-smi --query-gpu=pci.replay_counter --format=csv,noheader,nounits 2>/dev/null || echo "0")
    total_errors=0

    while IFS= read -r error; do
        total_errors=$((total_errors + error))
    done <<< "$pcie_errors"

    echo "  Total PCIe replay errors: $total_errors"

    if [ "$total_errors" -eq 0 ]; then
        add_result "PCIe" "PCIe_Errors" "pass" "0" "0"
    elif [ "$total_errors" -lt 100 ]; then
        add_result "PCIe" "PCIe_Errors" "warn" "$total_errors" "0" "Some PCIe errors detected"
    else
        add_result "PCIe" "PCIe_Errors" "fail" "$total_errors" "0" "High PCIe error count"
    fi
fi

# 4.3 PCIe ACS 检查
echo -e "\n${BLUE}4.3 PCIe ACS (Access Control Services)${NC}"
if dmesg | grep -q "ACS"; then
    echo "  ACS information found:"
    dmesg | grep "ACS" | tail -3 | sed 's/^/    /'
    add_result "PCIe" "PCIe_ACS" "pass" "present" "present"
else
    add_result "PCIe" "PCIe_ACS" "warn" "not detected" "present"
fi

#===========================================
# 5. GPU 配置检查
#===========================================
print_section "5. GPU Configuration"

if command -v nvidia-smi &> /dev/null; then
    # 5.1 GPU 基本信息
    echo -e "\n${BLUE}5.1 GPU Information${NC}"
    gpu_count=$(nvidia-smi -L | wc -l)
    driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)

    echo "  GPU Count: $gpu_count"
    echo "  Driver Version: $driver_version"

    add_result "GPU" "GPU_Count" "pass" "$gpu_count" ">0"
    add_result "GPU" "Driver_Version" "pass" "$driver_version" "installed"

    # 5.2 GPU 持久化模式
    echo -e "\n${BLUE}5.2 GPU Persistence Mode${NC}"
    persistence=$(nvidia-smi --query-gpu=persistence_mode --format=csv,noheader | head -1)

    if [[ "$persistence" == *"Enabled"* ]]; then
        add_result "GPU" "Persistence_Mode" "pass" "enabled" "enabled"
    else
        add_result "GPU" "Persistence_Mode" "warn" "disabled" "enabled" "Enable with: nvidia-smi -pm 1"
    fi

    # 5.3 GPU 温度
    echo -e "\n${BLUE}5.3 GPU Temperature${NC}"
    temps=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)
    max_temp=0

    while IFS= read -r temp; do
        echo "  GPU: ${temp}°C"
        if [ "$temp" -gt "$max_temp" ]; then
            max_temp=$temp
        fi
    done <<< "$temps"

    if [ "$max_temp" -lt 85 ]; then
        add_result "GPU" "GPU_Temperature" "pass" "${max_temp}°C" "<85°C"
    elif [ "$max_temp" -lt 95 ]; then
        add_result "GPU" "GPU_Temperature" "warn" "${max_temp}°C" "<85°C"
    else
        add_result "GPU" "GPU_Temperature" "fail" "${max_temp}°C" "<85°C"
    fi

    # 5.4 ECC 错误
    echo -e "\n${BLUE}5.4 ECC Memory Errors${NC}"
    ecc_errors=$(nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits 2>/dev/null || echo "N/A")

    if [[ "$ecc_errors" == *"N/A"* ]]; then
        add_result "GPU" "ECC_Errors" "pass" "not supported" "0 or N/A"
    else
        total_ecc=0
        while IFS= read -r err; do
            total_ecc=$((total_ecc + err))
        done <<< "$ecc_errors"

        if [ "$total_ecc" -eq 0 ]; then
            add_result "GPU" "ECC_Errors" "pass" "0" "0"
        else
            add_result "GPU" "ECC_Errors" "fail" "$total_ecc" "0"
        fi
    fi
else
    add_result "GPU" "NVIDIA_Driver" "fail" "not installed" "installed"
fi

#===========================================
# 6. 内存配置检查
#===========================================
print_section "6. Memory Configuration"

# 6.1 Transparent Huge Pages
echo -e "\n${BLUE}6.1 Transparent Huge Pages${NC}"
thp_status=$(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || echo "unknown")
echo "  THP Status: $thp_status"

if [[ "$thp_status" == *"[always]"* ]]; then
    add_result "Memory" "Transparent_Huge_Pages" "pass" "always" "always or madvise"
elif [[ "$thp_status" == *"[madvise]"* ]]; then
    add_result "Memory" "Transparent_Huge_Pages" "pass" "madvise" "always or madvise"
else
    add_result "Memory" "Transparent_Huge_Pages" "warn" "$thp_status" "always or madvise"
fi

# 6.2 Swappiness
echo -e "\n${BLUE}6.2 Swappiness${NC}"
swappiness=$(cat /proc/sys/vm/swappiness 2>/dev/null || echo "unknown")
echo "  Swappiness: $swappiness"

if [ "$swappiness" -le 10 ] 2>/dev/null; then
    add_result "Memory" "Swappiness" "pass" "$swappiness" "<=10"
else
    add_result "Memory" "Swappiness" "warn" "$swappiness" "<=10" "Lower value recommended for GPU workloads"
fi

# 6.3 总内存
echo -e "\n${BLUE}6.3 System Memory${NC}"
total_mem=$(free -g | awk '/^Mem:/ {print $2}')
echo "  Total Memory: ${total_mem}GB"
add_result "Memory" "Total_Memory" "pass" "${total_mem}GB" ">0"

#===========================================
# 7. 内核参数汇总
#===========================================
print_section "7. Kernel Parameters Summary"

echo -e "\n${BLUE}Current Kernel Command Line:${NC}"
echo "$cmdline" | fold -s -w 80 | sed 's/^/  /'

# 推荐的参数
echo -e "\n${BLUE}Recommended Additional Parameters:${NC}"
recommended_params=(
    "intel_iommu=on iommu=pt"
    "default_hugepagesz=1G hugepagesz=1G hugepages=32"
    "intel_idle.max_cstate=1 processor.max_cstate=1"
    "pcie_aspm=off"
)

for param in "${recommended_params[@]}"; do
    if echo "$cmdline" | grep -q "$(echo $param | awk '{print $1}')"; then
        echo -e "  ${GREEN}✓${NC} $param"
    else
        echo -e "  ${YELLOW}⚠${NC} Consider adding: $param"
    fi
done

#===========================================
# 8. 容器运行时检查 (可选)
#===========================================
print_section "8. Container Runtime (Optional)"

# 8.1 Docker
if command -v docker &> /dev/null; then
    docker_version=$(docker --version 2>/dev/null || echo "unknown")
    echo "  Docker: $docker_version"
    add_result "Container" "Docker" "pass" "installed" "installed"

    # 检查 NVIDIA Container Toolkit
    if docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi &>/dev/null; then
        add_result "Container" "NVIDIA_Container_Toolkit" "pass" "working" "working"
    else
        add_result "Container" "NVIDIA_Container_Toolkit" "warn" "not working" "working"
    fi
else
    add_result "Container" "Docker" "warn" "not installed" "installed (optional)"
fi

# 8.2 containerd
if command -v containerd &> /dev/null; then
    containerd_version=$(containerd --version 2>/dev/null | awk '{print $3}' || echo "unknown")
    echo "  containerd: $containerd_version"
    add_result "Container" "containerd" "pass" "installed" "installed"
else
    echo "  containerd: not installed"
fi

#===========================================
# 生成 JSON 报告
#===========================================
print_section "Generating Report"

# 计算总体状态
if [ $FAIL_COUNT -gt 0 ]; then
    OVERALL_STATUS="fail"
elif [ $WARN_COUNT -gt 0 ]; then
    OVERALL_STATUS="warn"
else
    OVERALL_STATUS="pass"
fi

# 生成 JSON
cat > "$OUTPUT_FILE" << EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "hostname": "$(hostname)",
  "overall_status": "$OVERALL_STATUS",
  "summary": {
    "total_checks": $((PASS_COUNT + WARN_COUNT + FAIL_COUNT)),
    "passed": $PASS_COUNT,
    "warnings": $WARN_COUNT,
    "failed": $FAIL_COUNT
  },
  "system_info": {
    "os": "$(lsb_release -d 2>/dev/null | cut -f2 || cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)",
    "kernel": "$(uname -r)",
    "cpu_model": "$(lscpu | grep 'Model name' | cut -d: -f2 | xargs)",
    "cpu_cores": "$(nproc)",
    "total_memory_gb": "$total_mem"
  },
  "checks": [
    $(IFS=,; echo "${RESULTS[*]}")
  ]
}
EOF

#===========================================
# 显示总结
#===========================================
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Validation Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Overall Status: $OVERALL_STATUS"
echo "Total Checks: $((PASS_COUNT + WARN_COUNT + FAIL_COUNT))"
echo -e "${GREEN}Passed: $PASS_COUNT${NC}"
echo -e "${YELLOW}Warnings: $WARN_COUNT${NC}"
echo -e "${RED}Failed: $FAIL_COUNT${NC}"
echo ""
echo "Detailed report saved to: $OUTPUT_FILE"
echo ""

# 退出码
if [ "$OVERALL_STATUS" = "fail" ]; then
    echo -e "${RED}System validation FAILED. Please review the issues above.${NC}"
    exit 1
elif [ "$OVERALL_STATUS" = "warn" ]; then
    echo -e "${YELLOW}System validation completed with warnings.${NC}"
    exit 0
else
    echo -e "${GREEN}All system checks PASSED!${NC}"
    exit 0
fi

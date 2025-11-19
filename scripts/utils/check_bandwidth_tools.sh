#!/bin/bash
################################################################################
# Bandwidth Testing Tools Dependency Checker
#
# Purpose: Check which bandwidth testing tools are installed and provide
#          installation instructions for missing ones
#
# Usage: ./check_bandwidth_tools.sh
################################################################################

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "========================================"
echo "Bandwidth Testing Tools Check"
echo "========================================"
echo ""

# Tool definitions
declare -A TOOLS=(
    ["nvbandwidth"]="NVIDIA bandwidth testing tool (official)"
    ["bandwidthTest"]="CUDA Samples PCIe bandwidth test"
    ["p2pBandwidthLatencyTest"]="CUDA Samples P2P bandwidth/latency test"
    ["ib_write_bw"]="RDMA/InfiniBand write bandwidth test (perftest)"
    ["ib_read_bw"]="RDMA/InfiniBand read bandwidth test (perftest)"
    ["ibstat"]="InfiniBand device status tool"
    ["ibv_devinfo"]="InfiniBand device information tool"
)

declare -A TOOL_SOURCES=(
    ["nvbandwidth"]="https://github.com/NVIDIA/nvbandwidth"
    ["bandwidthTest"]="https://github.com/NVIDIA/cuda-samples"
    ["p2pBandwidthLatencyTest"]="https://github.com/NVIDIA/cuda-samples"
    ["ib_write_bw"]="perftest package"
    ["ib_read_bw"]="perftest package"
    ["ibstat"]="infiniband-diags package"
    ["ibv_devinfo"]="infiniband-diags package"
)

# Check each tool
INSTALLED_COUNT=0
MISSING_COUNT=0
MISSING_TOOLS=()

echo -e "${BLUE}Checking installed tools...${NC}"
echo ""

for tool in "${!TOOLS[@]}"; do
    printf "%-30s " "$tool"
    if command -v "$tool" &> /dev/null; then
        echo -e "${GREEN}✓ Installed${NC} - $(which $tool)"
        INSTALLED_COUNT=$((INSTALLED_COUNT + 1))
    else
        echo -e "${RED}✗ Not Found${NC}"
        MISSING_TOOLS+=("$tool")
        MISSING_COUNT=$((MISSING_COUNT + 1))
    fi
done

echo ""
echo "========================================"
echo -e "Summary: ${GREEN}$INSTALLED_COUNT installed${NC}, ${RED}$MISSING_COUNT missing${NC}"
echo "========================================"
echo ""

# If tools are missing, provide installation instructions
if [ ${#MISSING_TOOLS[@]} -gt 0 ]; then
    echo -e "${YELLOW}Missing Tools and Installation Instructions:${NC}"
    echo ""

    # Group by installation method
    NEED_CUDA_SAMPLES=false
    NEED_NVBANDWIDTH=false
    NEED_PERFTEST=false
    NEED_IB_DIAGS=false

    for tool in "${MISSING_TOOLS[@]}"; do
        case $tool in
            bandwidthTest|p2pBandwidthLatencyTest)
                NEED_CUDA_SAMPLES=true
                ;;
            nvbandwidth)
                NEED_NVBANDWIDTH=true
                ;;
            ib_write_bw|ib_read_bw)
                NEED_PERFTEST=true
                ;;
            ibstat|ibv_devinfo)
                NEED_IB_DIAGS=true
                ;;
        esac
    done

    echo "Installation Methods:"
    echo ""

    # Method 1: Ansible (Recommended)
    echo -e "${GREEN}[Recommended] Use Ansible to install all tools:${NC}"
    echo "  cd ansible"
    echo "  ansible-playbook playbooks/install_benchmark_tools.yml"
    echo ""

    # Method 2: Individual installation
    echo -e "${BLUE}Or install individually:${NC}"
    echo ""

    if [ "$NEED_CUDA_SAMPLES" = true ]; then
        echo -e "${YELLOW}1. CUDA Samples (bandwidthTest, p2pBandwidthLatencyTest):${NC}"
        echo "   git clone https://github.com/NVIDIA/cuda-samples.git"
        echo "   cd cuda-samples/Samples/1_Utilities/bandwidthTest && make"
        echo "   cd ../p2pBandwidthLatencyTest && make"
        echo "   sudo cp bandwidthTest /usr/local/bin/"
        echo "   sudo cp p2pBandwidthLatencyTest /usr/local/bin/"
        echo ""
        echo "   Or use Ansible:"
        echo "   ansible-playbook playbooks/install_benchmark_tools.yml -t cuda_samples"
        echo ""
    fi

    if [ "$NEED_NVBANDWIDTH" = true ]; then
        echo -e "${YELLOW}2. nvbandwidth:${NC}"
        echo "   git clone https://github.com/NVIDIA/nvbandwidth.git"
        echo "   cd nvbandwidth && make"
        echo "   sudo cp nvbandwidth /usr/local/bin/"
        echo ""
        echo "   Or use Ansible:"
        echo "   ansible-playbook playbooks/install_benchmark_tools.yml -t nvbandwidth"
        echo ""
    fi

    if [ "$NEED_PERFTEST" = true ]; then
        echo -e "${YELLOW}3. perftest (RDMA bandwidth tests):${NC}"
        echo "   # Ubuntu/Debian:"
        echo "   sudo apt-get install perftest"
        echo ""
        echo "   # RHEL/CentOS/Rocky:"
        echo "   sudo yum install perftest"
        echo ""
        echo "   Or use Ansible:"
        echo "   ansible-playbook playbooks/install_benchmark_tools.yml -t perftest"
        echo ""
    fi

    if [ "$NEED_IB_DIAGS" = true ]; then
        echo -e "${YELLOW}4. InfiniBand diagnostics:${NC}"
        echo "   # Ubuntu/Debian:"
        echo "   sudo apt-get install infiniband-diags"
        echo ""
        echo "   # RHEL/CentOS/Rocky:"
        echo "   sudo yum install infiniband-diags"
        echo ""
    fi

    echo "========================================"
    echo ""
    echo -e "${BLUE}Note:${NC} All these tools are open source:"
    echo "  - nvbandwidth: Apache 2.0 License"
    echo "  - CUDA Samples: BSD 3-Clause License"
    echo "  - perftest: GPL/BSD License"
    echo ""
    echo "These are NOT proprietary/closed-source tools."
    echo "They just need to be installed separately."
    echo ""

else
    echo -e "${GREEN}✓ All bandwidth testing tools are installed!${NC}"
    echo ""
    echo "You can now run:"
    echo "  ./scripts/validation/bandwidth_test.sh"
    echo "  ./scripts/validation/intra_node_bandwidth_check.sh"
    echo ""
fi

# Check for CUDA Toolkit
echo "========================================"
echo "Additional Information"
echo "========================================"
echo ""

if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | sed 's/,//')
    echo -e "CUDA Toolkit: ${GREEN}✓ Installed${NC} (Version: $CUDA_VERSION)"
else
    echo -e "CUDA Toolkit: ${RED}✗ Not Found${NC}"
    echo "  CUDA Toolkit is required to compile CUDA Samples"
fi

if command -v nvidia-smi &> /dev/null; then
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    echo -e "NVIDIA Driver: ${GREEN}✓ Installed${NC} (Version: $DRIVER_VERSION)"
else
    echo -e "NVIDIA Driver: ${RED}✗ Not Found${NC}"
fi

echo ""
echo "For detailed information about these tools, see:"
echo "  docs/bandwidth_commands_analysis.md"
echo ""

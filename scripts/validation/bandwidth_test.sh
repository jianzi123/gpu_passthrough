#!/bin/bash
# Comprehensive Bandwidth Testing Script
# Tests PCIe, NVLink, and RDMA bandwidth
# Compares against performance baselines

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Output file
OUTPUT_DIR="${1:-/tmp/bandwidth_tests}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_JSON="$OUTPUT_DIR/bandwidth_test_${TIMESTAMP}.json"

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "GPU Bandwidth Testing Suite"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Timestamp: $TIMESTAMP"
echo ""

# Detect GPU model
GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -d ' ')
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)

echo "Detected GPU: $GPU_MODEL"
echo "GPU Count: $GPU_COUNT"
echo ""

# Initialize results
RESULTS='{'
RESULTS+='"timestamp":"'$(date -u +%Y-%m-%dT%H:%M:%SZ)'",'
RESULTS+='"gpu_model":"'$GPU_MODEL'",'
RESULTS+='"gpu_count":'$GPU_COUNT','
RESULTS+='"tests":{'

#==========================================
# Test 1: PCIe Bandwidth
#==========================================
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Test 1: PCIe Bandwidth${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

PCIE_RESULTS='{"name":"PCIe Bandwidth",'

# Method 1: Using nvbandwidth (if available)
if command -v nvbandwidth &> /dev/null; then
    echo "Using nvbandwidth..."
    nvbandwidth -t host_to_device_memcpy_ce,device_to_host_memcpy_ce > "$OUTPUT_DIR/nvbandwidth_${TIMESTAMP}.txt" 2>&1 || true

    # Parse results
    H2D_BW=$(grep -A 5 "host_to_device_memcpy_ce" "$OUTPUT_DIR/nvbandwidth_${TIMESTAMP}.txt" | grep "SUM" | awk '{print $2}' || echo "0")
    D2H_BW=$(grep -A 5 "device_to_host_memcpy_ce" "$OUTPUT_DIR/nvbandwidth_${TIMESTAMP}.txt" | grep "SUM" | awk '{print $2}' || echo "0")

    PCIE_RESULTS+='"nvbandwidth":{'
    PCIE_RESULTS+='"host_to_device_gbs":'${H2D_BW:-0}','
    PCIE_RESULTS+='"device_to_host_gbs":'${D2H_BW:-0}
    PCIE_RESULTS+='},'

    echo "  Host to Device: ${H2D_BW} GB/s"
    echo "  Device to Host: ${D2H_BW} GB/s"
fi

# Method 2: Using bandwidthTest from CUDA samples (if available)
if command -v bandwidthTest &> /dev/null; then
    echo ""
    echo "Using bandwidthTest (CUDA Samples)..."
    bandwidthTest --htod --dtoh > "$OUTPUT_DIR/bandwidthTest_${TIMESTAMP}.txt" 2>&1 || true

    # Parse results
    HTOD_BW=$(grep "Host to Device" "$OUTPUT_DIR/bandwidthTest_${TIMESTAMP}.txt" | tail -1 | awk '{print $(NF-1)}' || echo "0")
    DTOH_BW=$(grep "Device to Host" "$OUTPUT_DIR/bandwidthTest_${TIMESTAMP}.txt" | tail -1 | awk '{print $(NF-1)}' || echo "0")

    # Convert MB/s to GB/s
    HTOD_GB=$(echo "scale=2; $HTOD_BW / 1000" | bc)
    DTOH_GB=$(echo "scale=2; $DTOH_BW / 1000" | bc)

    PCIE_RESULTS+='"bandwidthTest":{'
    PCIE_RESULTS+='"host_to_device_gbs":'${HTOD_GB:-0}','
    PCIE_RESULTS+='"device_to_host_gbs":'${DTOH_GB:-0}
    PCIE_RESULTS+='}'

    echo "  Host to Device: ${HTOD_GB} GB/s"
    echo "  Device to Host: ${DTOH_GB} GB/s"
fi

# Check PCIe link status
echo ""
echo "PCIe Link Status:"
nvidia-smi --query-gpu=index,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max --format=csv

PCIE_RESULTS+='}'
RESULTS+="\"pcie\":$PCIE_RESULTS,"

#==========================================
# Test 2: NVLink Bandwidth (if available)
#==========================================
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Test 2: NVLink Bandwidth${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

NVLINK_RESULTS='{"name":"NVLink Bandwidth",'

# Check if NVLink is available
if nvidia-smi nvlink --status &> /dev/null; then
    echo "NVLink detected:"
    nvidia-smi topo -m

    # Using p2pBandwidthLatencyTest (if available)
    if command -v p2pBandwidthLatencyTest &> /dev/null; then
        echo ""
        echo "Running p2pBandwidthLatencyTest..."
        p2pBandwidthLatencyTest > "$OUTPUT_DIR/p2p_test_${TIMESTAMP}.txt" 2>&1 || true

        # Parse results - get P2P bandwidth
        P2P_BW=$(grep "Unidirectional" "$OUTPUT_DIR/p2p_test_${TIMESTAMP}.txt" | head -1 | awk '{print $NF}' || echo "0")
        BIDIR_BW=$(grep "Bidirectional" "$OUTPUT_DIR/p2p_test_${TIMESTAMP}.txt" | head -1 | awk '{print $NF}' || echo "0")

        NVLINK_RESULTS+='"p2p_unidirectional_gbs":'${P2P_BW:-0}','
        NVLINK_RESULTS+='"p2p_bidirectional_gbs":'${BIDIR_BW:-0}','

        echo "  Unidirectional: ${P2P_BW} GB/s"
        echo "  Bidirectional: ${BIDIR_BW} GB/s"
    fi

    # Using nvbandwidth for NVLink
    if command -v nvbandwidth &> /dev/null; then
        echo ""
        echo "Running nvbandwidth for GPU-to-GPU..."
        nvbandwidth -t device_to_device_memcpy_read_ce,device_to_device_memcpy_write_ce -s $((32 * 1024 * 1024)) > "$OUTPUT_DIR/nvbandwidth_nvlink_${TIMESTAMP}.txt" 2>&1 || true

        # Parse GPU-to-GPU bandwidth
        D2D_BW=$(grep -A 10 "device_to_device" "$OUTPUT_DIR/nvbandwidth_nvlink_${TIMESTAMP}.txt" | grep "SUM" | awk '{print $2}' | head -1 || echo "0")

        NVLINK_RESULTS+='"device_to_device_gbs":'${D2D_BW:-0}

        echo "  Device to Device: ${D2D_BW} GB/s"
    fi

    NVLINK_RESULTS+=',"available":true}'
else
    echo "NVLink not available on this system"
    NVLINK_RESULTS+=',"available":false}'
fi

RESULTS+="\"nvlink\":$NVLINK_RESULTS,"

#==========================================
# Test 3: RDMA Bandwidth (if available)
#==========================================
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Test 3: RDMA Bandwidth${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

RDMA_RESULTS='{"name":"RDMA Bandwidth",'

# Check if InfiniBand/RDMA is available
if command -v ibstat &> /dev/null; then
    echo "InfiniBand detected:"
    ibstat | grep -E "State|Rate|Link layer"

    IB_DEVICE=$(ibstat -l | head -1)
    echo "Using IB device: $IB_DEVICE"

    # Check if ib_write_bw is available
    if command -v ib_write_bw &> /dev/null; then
        echo ""
        echo "RDMA bandwidth test requires a remote host."
        echo "To test RDMA, run on two nodes:"
        echo "  Node 1 (server): ib_write_bw -d $IB_DEVICE"
        echo "  Node 2 (client): ib_write_bw -d $IB_DEVICE <server_ip>"
        echo ""
        echo "For GPUDirect RDMA (if CUDA-enabled perftest):"
        echo "  Server: ib_write_bw -d $IB_DEVICE --use_cuda=0"
        echo "  Client: ib_write_bw -d $IB_DEVICE --use_cuda=0 <server_ip>"

        # Check if we can get device info
        IB_RATE=$(ibstat | grep "Rate:" | head -1 | awk '{print $2}')
        RDMA_RESULTS+='"ib_device":"'$IB_DEVICE'",'
        RDMA_RESULTS+='"ib_rate":"'${IB_RATE:-unknown}'",'
        RDMA_RESULTS+='"available":true,'
        RDMA_RESULTS+='"note":"Manual testing required"'
    else
        echo "ib_write_bw not found. Install perftest package."
        RDMA_RESULTS+='"available":false,'
        RDMA_RESULTS+='"note":"perftest not installed"'
    fi

    RDMA_RESULTS+='}'
elif ip link show | grep -q "ib"; then
    echo "RoCE (RDMA over Converged Ethernet) may be available"
    RDMA_RESULTS+='"available":"maybe",'
    RDMA_RESULTS+='"note":"RoCE detected, install perftest for testing"}'
else
    echo "No RDMA devices detected"
    RDMA_RESULTS+='"available":false}'
fi

RESULTS+="\"rdma\":$RDMA_RESULTS"

#==========================================
# Close results JSON
#==========================================
RESULTS+='}}'

# Write JSON output
echo "$RESULTS" | python3 -m json.tool > "$OUTPUT_JSON" 2>/dev/null || echo "$RESULTS" > "$OUTPUT_JSON"

#==========================================
# Compare against baselines
#==========================================
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Baseline Comparison${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Load baseline script if available
BASELINE_SCRIPT="$(dirname $0)/../utils/performance_baselines.py"
if [ -f "$BASELINE_SCRIPT" ]; then
    python3 "$BASELINE_SCRIPT" info "$GPU_MODEL" 2>/dev/null || echo "Baseline data not available for $GPU_MODEL"
else
    echo "Baseline script not found: $BASELINE_SCRIPT"
fi

#==========================================
# Summary
#==========================================
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Test Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Results saved to:"
echo "  JSON: $OUTPUT_JSON"
echo "  Logs: $OUTPUT_DIR/*_${TIMESTAMP}.txt"
echo ""
echo "To view full results:"
echo "  cat $OUTPUT_JSON | python3 -m json.tool"
echo ""

# Check for performance issues
echo "Performance Check:"
if [ -n "${H2D_BW:-}" ] && [ "${H2D_BW:-0}" != "0" ]; then
    if (( $(echo "$H2D_BW < 10" | bc -l) )); then
        echo -e "  ${RED}⚠${NC} Low PCIe bandwidth detected (${H2D_BW} GB/s)"
    else
        echo -e "  ${GREEN}✓${NC} PCIe bandwidth OK (${H2D_BW} GB/s)"
    fi
fi

if [ -n "${P2P_BW:-}" ] && [ "${P2P_BW:-0}" != "0" ]; then
    if (( $(echo "$P2P_BW < 100" | bc -l) )); then
        echo -e "  ${YELLOW}⚠${NC} Lower than expected NVLink bandwidth (${P2P_BW} GB/s)"
    else
        echo -e "  ${GREEN}✓${NC} NVLink bandwidth OK (${P2P_BW} GB/s)"
    fi
fi

echo ""
echo "Bandwidth testing complete!"

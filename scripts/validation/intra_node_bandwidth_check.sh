#!/bin/bash
################################################################################
# Intra-Node Bandwidth Checker
#
# Purpose: Detect slow GPUs within a node by checking:
#   - NVLink topology and status
#   - PCIe bandwidth
#   - GPU-to-GPU bandwidth (NVLink and PCIe paths)
#   - Comparison against performance baselines
#
# Usage: ./intra_node_bandwidth_check.sh [options]
#
# Options:
#   -o, --output DIR       Output directory for results (default: ./bandwidth_check_results)
#   -b, --baseline FILE    Custom baseline file (default: use built-in baselines)
#   -t, --threshold PCT    Performance threshold percentage (default: 90)
#   -v, --verbose          Verbose output
#   -h, --help             Show this help message
#
# Dependencies:
#   - nvidia-smi
#   - nvbandwidth (from CUDA samples)
#   - bandwidthTest (from CUDA samples)
#   - p2pBandwidthLatencyTest (from CUDA samples)
#
################################################################################

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
OUTPUT_DIR="./bandwidth_check_results"
BASELINE_FILE=""
THRESHOLD=90  # Percentage of expected baseline
VERBOSE=0
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Performance baselines (GB/s)
declare -A NVLINK_BASELINES=(
    ["A100-SXM4"]="600"      # NVLink 3.0: 600 GB/s total per GPU
    ["A100-PCIE"]="0"        # No NVLink on PCIe version
    ["H100-SXM5"]="900"      # NVLink 4.0: 900 GB/s total per GPU
    ["H100-PCIE"]="0"        # No NVLink on PCIe version
    ["A800"]="400"           # NVLink 3.0 restricted
    ["H800"]="900"           # NVLink 4.0
    ["V100-SXM2"]="300"      # NVLink 2.0: 300 GB/s total per GPU
    ["V100-PCIE"]="0"        # No NVLink on PCIe version
)

declare -A PCIE_BASELINES=(
    ["PCIE-GEN3-X16"]="15.75"   # PCIe Gen3 x16: ~16 GB/s
    ["PCIE-GEN4-X16"]="31.5"    # PCIe Gen4 x16: ~32 GB/s
    ["PCIE-GEN5-X16"]="63"      # PCIe Gen5 x16: ~64 GB/s
)

# Function to print colored output
print_color() {
    local color=$1
    shift
    echo -e "${color}$@${NC}"
}

# Function to print verbose output
verbose() {
    if [ $VERBOSE -eq 1 ]; then
        print_color "$BLUE" "[VERBOSE] $@"
    fi
}

# Function to show usage
show_usage() {
    grep '^#' "$0" | grep -v '#!/bin/bash' | sed 's/^# \?//'
    exit 0
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -o|--output)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            -b|--baseline)
                BASELINE_FILE="$2"
                shift 2
                ;;
            -t|--threshold)
                THRESHOLD="$2"
                shift 2
                ;;
            -v|--verbose)
                VERBOSE=1
                shift
                ;;
            -h|--help)
                show_usage
                ;;
            *)
                echo "Unknown option: $1"
                show_usage
                ;;
        esac
    done
}

# Function to check dependencies
check_dependencies() {
    print_color "$BLUE" "=== Checking Dependencies ==="

    local missing_deps=0

    if ! command -v nvidia-smi &> /dev/null; then
        print_color "$RED" "ERROR: nvidia-smi not found"
        missing_deps=1
    fi

    # Check for CUDA bandwidth test tools
    local cuda_bin_paths=(
        "/usr/local/cuda/extras/demo_suite"
        "/usr/local/cuda/samples/bin"
        "/usr/local/cuda/bin"
        "$HOME/cuda-samples/bin"
    )

    local found_bandwidth_test=0
    for path in "${cuda_bin_paths[@]}"; do
        if [ -f "$path/bandwidthTest" ]; then
            BANDWIDTH_TEST="$path/bandwidthTest"
            found_bandwidth_test=1
            break
        fi
    done

    if [ $found_bandwidth_test -eq 0 ]; then
        print_color "$YELLOW" "WARNING: bandwidthTest not found in standard CUDA paths"
        print_color "$YELLOW" "         Will skip bandwidth tests requiring this tool"
        BANDWIDTH_TEST=""
    fi

    local found_p2p_test=0
    for path in "${cuda_bin_paths[@]}"; do
        if [ -f "$path/p2pBandwidthLatencyTest" ]; then
            P2P_TEST="$path/p2pBandwidthLatencyTest"
            found_p2p_test=1
            break
        fi
    done

    if [ $found_p2p_test -eq 0 ]; then
        print_color "$YELLOW" "WARNING: p2pBandwidthLatencyTest not found"
        print_color "$YELLOW" "         Will use alternative methods for P2P testing"
        P2P_TEST=""
    fi

    # Check for nvbandwidth (newer tool)
    if command -v nvbandwidth &> /dev/null; then
        NVBANDWIDTH="nvbandwidth"
        verbose "Found nvbandwidth: $(which nvbandwidth)"
    else
        print_color "$YELLOW" "WARNING: nvbandwidth not found"
        NVBANDWIDTH=""
    fi

    if [ $missing_deps -eq 1 ]; then
        print_color "$RED" "ERROR: Missing required dependencies"
        exit 1
    fi

    print_color "$GREEN" "✓ Dependency check passed"
    echo
}

# Function to get GPU information
get_gpu_info() {
    print_color "$BLUE" "=== Gathering GPU Information ==="

    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
    echo "GPU Count: $GPU_COUNT"

    # Get GPU model
    GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo "GPU Model: $GPU_MODEL"

    # Detect NVLink capability
    NVLINK_CAPABLE=0
    if nvidia-smi nvlink --status &>/dev/null; then
        NVLINK_CAPABLE=1
        echo "NVLink: Capable"
    else
        echo "NVLink: Not Available"
    fi

    # Save GPU info to file
    mkdir -p "$OUTPUT_DIR"
    cat > "$OUTPUT_DIR/gpu_info_${TIMESTAMP}.txt" <<EOF
GPU Count: $GPU_COUNT
GPU Model: $GPU_MODEL
NVLink Capable: $NVLINK_CAPABLE
Timestamp: $(date)
EOF

    echo
}

# Function to check NVLink topology and status
check_nvlink_topology() {
    if [ $NVLINK_CAPABLE -eq 0 ]; then
        print_color "$YELLOW" "Skipping NVLink topology check (not available)"
        return
    fi

    print_color "$BLUE" "=== Checking NVLink Topology ==="

    local output_file="$OUTPUT_DIR/nvlink_topology_${TIMESTAMP}.txt"

    # Get NVLink status for each GPU
    for i in $(seq 0 $((GPU_COUNT - 1))); do
        echo "GPU $i NVLink Status:" >> "$output_file"
        nvidia-smi nvlink --status -i $i >> "$output_file" 2>&1 || true
        echo "" >> "$output_file"
    done

    # Check for inactive links
    local inactive_links=$(grep -i "inactive\|down" "$output_file" || true)
    if [ -n "$inactive_links" ]; then
        print_color "$RED" "⚠ WARNING: Found inactive NVLink connections!"
        echo "$inactive_links"
    else
        print_color "$GREEN" "✓ All NVLink connections are active"
    fi

    # Get NVLink topology
    echo "=== NVLink Topology ===" >> "$output_file"
    nvidia-smi topo -m >> "$output_file" 2>&1 || true

    verbose "NVLink topology saved to: $output_file"
    echo
}

# Function to test GPU-to-GPU bandwidth using p2pBandwidthLatencyTest
test_p2p_bandwidth() {
    print_color "$BLUE" "=== Testing GPU-to-GPU P2P Bandwidth ==="

    local output_file="$OUTPUT_DIR/p2p_bandwidth_${TIMESTAMP}.txt"
    local summary_file="$OUTPUT_DIR/p2p_bandwidth_summary_${TIMESTAMP}.csv"

    echo "GPU_Pair,Bandwidth_GB/s,Connection_Type" > "$summary_file"

    if [ -n "$P2P_TEST" ]; then
        # Use NVIDIA's p2pBandwidthLatencyTest
        verbose "Running p2pBandwidthLatencyTest..."
        $P2P_TEST > "$output_file" 2>&1 || true

        # Parse results and create summary
        parse_p2p_results "$output_file" "$summary_file"
    elif [ -n "$NVBANDWIDTH" ]; then
        # Use nvbandwidth as alternative
        verbose "Running nvbandwidth for P2P testing..."
        $NVBANDWIDTH -t device_to_device > "$output_file" 2>&1 || true

        # Parse nvbandwidth results
        parse_nvbandwidth_results "$output_file" "$summary_file"
    else
        print_color "$YELLOW" "No P2P bandwidth test tool available, using nvidia-smi only"
        nvidia-smi topo -p2p w > "$output_file" 2>&1 || true
    fi

    # Analyze results for slow connections
    analyze_p2p_bandwidth "$summary_file"

    echo
}

# Function to parse p2pBandwidthLatencyTest results
parse_p2p_results() {
    local input_file=$1
    local output_file=$2

    # Extract bidirectional bandwidth from p2pBandwidthLatencyTest output
    # Example line: "Bidirectional P2P=Enabled Bandwidth Matrix (GB/s)"

    if grep -q "Bidirectional.*Bandwidth Matrix" "$input_file"; then
        # Parse the bandwidth matrix
        awk '/Bidirectional.*Bandwidth Matrix/,/^$/ {
            if ($1 ~ /^[0-9]+$/ && NF > 1) {
                gpu_id = $1
                for (i = 2; i <= NF; i++) {
                    if ($i != "N/A" && $i ~ /^[0-9.]+$/) {
                        peer_id = i - 2
                        if (gpu_id < peer_id) {  # Avoid duplicates
                            print gpu_id "-" peer_id "," $i ",Unknown"
                        }
                    }
                }
            }
        }' "$input_file" >> "$output_file"
    fi
}

# Function to parse nvbandwidth results
parse_nvbandwidth_results() {
    local input_file=$1
    local output_file=$2

    # Parse nvbandwidth output
    # This is a simplified parser - adjust based on actual nvbandwidth output format
    grep -E "GPU [0-9]+ -> GPU [0-9]+" "$input_file" | \
    awk '{print $2 "-" $5 "," $7 ",Unknown"}' >> "$output_file" 2>/dev/null || true
}

# Function to analyze P2P bandwidth and detect slow connections
analyze_p2p_bandwidth() {
    local summary_file=$1

    if [ ! -s "$summary_file" ] || [ $(wc -l < "$summary_file") -le 1 ]; then
        print_color "$YELLOW" "No P2P bandwidth data to analyze"
        return
    fi

    print_color "$BLUE" "=== Analyzing P2P Bandwidth ==="

    # Calculate statistics
    local avg_bandwidth=$(awk -F',' 'NR>1 {sum+=$2; count++} END {if(count>0) print sum/count; else print 0}' "$summary_file")
    local min_bandwidth=$(awk -F',' 'NR>1 {if(min=="" || $2<min) min=$2} END {print min}' "$summary_file")
    local max_bandwidth=$(awk -F',' 'NR>1 {if(max=="" || $2>max) max=$2} END {print max}' "$summary_file")

    echo "Average Bandwidth: ${avg_bandwidth} GB/s"
    echo "Min Bandwidth: ${min_bandwidth} GB/s"
    echo "Max Bandwidth: ${max_bandwidth} GB/s"

    # Detect expected bandwidth based on GPU model
    local expected_bandwidth=0
    if [[ "$GPU_MODEL" =~ "A100" ]] && [ $NVLINK_CAPABLE -eq 1 ]; then
        if [[ "$GPU_MODEL" =~ "SXM4" ]] || [[ "$GPU_MODEL" =~ "SXM" ]]; then
            expected_bandwidth=300  # Per-link NVLink 3.0 bandwidth
        fi
    elif [[ "$GPU_MODEL" =~ "H100" ]] && [ $NVLINK_CAPABLE -eq 1 ]; then
        expected_bandwidth=450  # Per-link NVLink 4.0 bandwidth
    elif [[ "$GPU_MODEL" =~ "V100" ]] && [ $NVLINK_CAPABLE -eq 1 ]; then
        expected_bandwidth=150  # Per-link NVLink 2.0 bandwidth
    fi

    if [ "$expected_bandwidth" -gt 0 ]; then
        local threshold_bandwidth=$(echo "$expected_bandwidth * $THRESHOLD / 100" | bc -l)
        echo "Expected Bandwidth: ${expected_bandwidth} GB/s"
        echo "Threshold (${THRESHOLD}%): ${threshold_bandwidth} GB/s"

        # Find slow connections
        local slow_connections=$(awk -F',' -v thresh="$threshold_bandwidth" 'NR>1 && $2 < thresh {print}' "$summary_file")

        if [ -n "$slow_connections" ]; then
            print_color "$RED" "⚠ WARNING: Found slow GPU-to-GPU connections:"
            echo "$slow_connections" | while read line; do
                echo "  $line"
            done

            # Save slow connections to separate file
            echo "$slow_connections" > "$OUTPUT_DIR/slow_connections_${TIMESTAMP}.txt"
        else
            print_color "$GREEN" "✓ All GPU-to-GPU connections meet performance threshold"
        fi
    else
        print_color "$YELLOW" "No baseline available for $GPU_MODEL, skipping threshold check"
    fi

    echo
}

# Function to test PCIe bandwidth
test_pcie_bandwidth() {
    print_color "$BLUE" "=== Testing PCIe Bandwidth ==="

    local output_file="$OUTPUT_DIR/pcie_bandwidth_${TIMESTAMP}.txt"

    if [ -n "$BANDWIDTH_TEST" ]; then
        verbose "Running bandwidthTest for each GPU..."

        for i in $(seq 0 $((GPU_COUNT - 1))); do
            echo "=== GPU $i PCIe Bandwidth ===" >> "$output_file"
            CUDA_VISIBLE_DEVICES=$i $BANDWIDTH_TEST --htod --dtoh >> "$output_file" 2>&1 || true
            echo "" >> "$output_file"
        done

        # Parse and analyze results
        analyze_pcie_bandwidth "$output_file"
    else
        print_color "$YELLOW" "bandwidthTest not available, checking PCIe link info only"

        for i in $(seq 0 $((GPU_COUNT - 1))); do
            echo "GPU $i:" >> "$output_file"
            nvidia-smi -i $i --query-gpu=pci.bus_id,pcie.link.gen.current,pcie.link.width.current --format=csv >> "$output_file"
        done

        cat "$output_file"
    fi

    echo
}

# Function to analyze PCIe bandwidth
analyze_pcie_bandwidth() {
    local output_file=$1

    print_color "$BLUE" "=== Analyzing PCIe Bandwidth ==="

    # Extract Host to Device and Device to Host bandwidth for each GPU
    local summary_file="$OUTPUT_DIR/pcie_bandwidth_summary_${TIMESTAMP}.csv"
    echo "GPU,HtoD_GB/s,DtoH_GB/s,PCIe_Gen,PCIe_Width" > "$summary_file"

    local current_gpu=0
    while IFS= read -r line; do
        if [[ "$line" =~ "GPU $current_gpu PCIe Bandwidth" ]]; then
            # Extract bandwidth values from subsequent lines
            local htod=$(grep "Host to Device" "$output_file" | awk -v gpu=$current_gpu 'NR==gpu+1 {print $(NF-1)}')
            local dtoh=$(grep "Device to Host" "$output_file" | awk -v gpu=$current_gpu 'NR==gpu+1 {print $(NF-1)}')

            # Get PCIe generation and width
            local pcie_info=$(nvidia-smi -i $current_gpu --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv,noheader)
            local pcie_gen=$(echo "$pcie_info" | cut -d',' -f1 | tr -d ' ')
            local pcie_width=$(echo "$pcie_info" | cut -d',' -f2 | tr -d ' ')

            echo "$current_gpu,$htod,$dtoh,$pcie_gen,$pcie_width" >> "$summary_file"
            current_gpu=$((current_gpu + 1))
        fi
    done < "$output_file"

    # Display summary
    column -t -s',' "$summary_file"

    # Check for anomalies
    local issues_found=0

    while IFS=, read -r gpu htod dtoh pcie_gen pcie_width; do
        if [ "$gpu" = "GPU" ]; then continue; fi  # Skip header

        # Check if PCIe link is at expected width (usually x16)
        if [ "$pcie_width" != "16" ] && [ -n "$pcie_width" ]; then
            print_color "$YELLOW" "⚠ GPU $gpu: PCIe width is x${pcie_width} (expected x16)"
            issues_found=1
        fi

        # Check bandwidth against expected for PCIe generation
        if [ -n "$pcie_gen" ] && [ -n "$htod" ]; then
            local expected_key="PCIE-GEN${pcie_gen}-X${pcie_width}"
            local expected_bw=${PCIE_BASELINES[$expected_key]:-0}

            if [ "$expected_bw" != "0" ]; then
                local threshold_bw=$(echo "$expected_bw * $THRESHOLD / 100" | bc -l)
                local htod_check=$(echo "$htod < $threshold_bw" | bc -l)

                if [ "$htod_check" -eq 1 ]; then
                    print_color "$RED" "⚠ GPU $gpu: PCIe HtoD bandwidth ($htod GB/s) below threshold ($threshold_bw GB/s)"
                    issues_found=1
                fi
            fi
        fi
    done < "$summary_file"

    if [ $issues_found -eq 0 ]; then
        print_color "$GREEN" "✓ All GPUs show normal PCIe performance"
    fi

    echo
}

# Function to generate comprehensive report
generate_report() {
    print_color "$BLUE" "=== Generating Comprehensive Report ==="

    local report_file="$OUTPUT_DIR/bandwidth_check_report_${TIMESTAMP}.md"

    cat > "$report_file" <<EOF
# Intra-Node Bandwidth Check Report

**Timestamp:** $(date)
**Hostname:** $(hostname)
**GPU Count:** $GPU_COUNT
**GPU Model:** $GPU_MODEL
**NVLink Capable:** $NVLINK_CAPABLE
**Threshold:** ${THRESHOLD}%

---

## Summary

EOF

    # Add GPU info
    echo "### GPU Configuration" >> "$report_file"
    echo '```' >> "$report_file"
    nvidia-smi --query-gpu=index,name,pci.bus_id,pcie.link.gen.current,pcie.link.width.current --format=csv >> "$report_file"
    echo '```' >> "$report_file"
    echo "" >> "$report_file"

    # Add NVLink topology if available
    if [ $NVLINK_CAPABLE -eq 1 ]; then
        echo "### NVLink Topology" >> "$report_file"
        echo '```' >> "$report_file"
        nvidia-smi topo -m >> "$report_file"
        echo '```' >> "$report_file"
        echo "" >> "$report_file"
    fi

    # Add P2P bandwidth summary if available
    if [ -f "$OUTPUT_DIR/p2p_bandwidth_summary_${TIMESTAMP}.csv" ]; then
        echo "### GPU-to-GPU Bandwidth" >> "$report_file"
        echo '```' >> "$report_file"
        cat "$OUTPUT_DIR/p2p_bandwidth_summary_${TIMESTAMP}.csv" >> "$report_file"
        echo '```' >> "$report_file"
        echo "" >> "$report_file"
    fi

    # Add PCIe bandwidth summary if available
    if [ -f "$OUTPUT_DIR/pcie_bandwidth_summary_${TIMESTAMP}.csv" ]; then
        echo "### PCIe Bandwidth" >> "$report_file"
        echo '```' >> "$report_file"
        cat "$OUTPUT_DIR/pcie_bandwidth_summary_${TIMESTAMP}.csv" >> "$report_file"
        echo '```' >> "$report_file"
        echo "" >> "$report_file"
    fi

    # Add slow connections if found
    if [ -f "$OUTPUT_DIR/slow_connections_${TIMESTAMP}.txt" ]; then
        echo "### ⚠ Issues Detected" >> "$report_file"
        echo "" >> "$report_file"
        echo "**Slow GPU-to-GPU Connections:**" >> "$report_file"
        echo '```' >> "$report_file"
        cat "$OUTPUT_DIR/slow_connections_${TIMESTAMP}.txt" >> "$report_file"
        echo '```' >> "$report_file"
        echo "" >> "$report_file"
    fi

    # Add recommendations
    echo "### Recommendations" >> "$report_file"
    echo "" >> "$report_file"

    if [ -f "$OUTPUT_DIR/slow_connections_${TIMESTAMP}.txt" ]; then
        echo "- Investigate GPUs with bandwidth below threshold" >> "$report_file"
        echo "- Check NVLink cable connections" >> "$report_file"
        echo "- Verify NVIDIA driver and firmware versions" >> "$report_file"
        echo "- Check for hardware faults" >> "$report_file"
    else
        echo "- All bandwidth tests passed successfully" >> "$report_file"
        echo "- System is performing within expected parameters" >> "$report_file"
    fi

    echo "" >> "$report_file"
    echo "---" >> "$report_file"
    echo "Report generated by: intra_node_bandwidth_check.sh" >> "$report_file"

    print_color "$GREEN" "✓ Report saved to: $report_file"
    echo
}

# Main execution
main() {
    parse_args "$@"

    print_color "$GREEN" "========================================"
    print_color "$GREEN" "   Intra-Node Bandwidth Check"
    print_color "$GREEN" "========================================"
    echo

    check_dependencies
    get_gpu_info
    check_nvlink_topology
    test_p2p_bandwidth
    test_pcie_bandwidth
    generate_report

    print_color "$GREEN" "========================================"
    print_color "$GREEN" "   Bandwidth Check Complete"
    print_color "$GREEN" "========================================"
    print_color "$BLUE" "Results saved to: $OUTPUT_DIR"
    echo
}

# Run main function
main "$@"

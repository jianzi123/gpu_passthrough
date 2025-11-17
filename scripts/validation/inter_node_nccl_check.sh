#!/bin/bash
################################################################################
# Inter-Node NCCL Communication Checker
#
# Purpose: Detect slow nodes in a GPU cluster by:
#   - Running multiple NCCL all-reduce tests across nodes
#   - Collecting statistical data (mean, stddev, min, max)
#   - Using binary search to isolate problematic nodes
#   - Performing pairwise node testing
#   - Comparing against performance baselines
#
# Usage: ./inter_node_nccl_check.sh [options]
#
# Options:
#   -n, --nodes FILE          File containing list of nodes (one per line)
#   -g, --gpus-per-node N     Number of GPUs per node (default: auto-detect)
#   -i, --iterations N        Number of test iterations (default: 10)
#   -s, --size SIZE           Message size for all-reduce (default: 8G)
#   -o, --output DIR          Output directory for results
#   -t, --threshold PCT       Performance threshold percentage (default: 92)
#   -b, --baseline FILE       Custom baseline file
#   --mpi-path PATH           Path to MPI installation (default: auto-detect)
#   --nccl-tests-path PATH    Path to nccl-tests binaries (default: auto-detect)
#   --pairwise                Enable pairwise node testing
#   --binary-search           Enable binary search for slow node detection
#   -v, --verbose             Verbose output
#   -h, --help                Show this help message
#
# Dependencies:
#   - MPI (OpenMPI or MPICH)
#   - nccl-tests (https://github.com/NVIDIA/nccl-tests)
#   - nvidia-smi on all nodes
#   - SSH access to all nodes
#
# Environment Variables:
#   - NCCL_DEBUG: Set to INFO or TRACE for detailed NCCL logging
#   - NCCL_IB_DISABLE: Set to 1 to disable InfiniBand
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
NODES_FILE=""
GPUS_PER_NODE=0  # Auto-detect
ITERATIONS=10
MESSAGE_SIZE="8G"
OUTPUT_DIR="./nccl_check_results"
THRESHOLD=92  # NCCL tests typically achieve ~92% of theoretical bandwidth
BASELINE_FILE=""
MPI_PATH=""
NCCL_TESTS_PATH=""
PAIRWISE=0
BINARY_SEARCH=0
VERBOSE=0
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Performance baselines (GB/s bus bandwidth for all-reduce)
declare -A NCCL_BASELINES=(
    # Intra-node (8 GPUs)
    ["A100-SXM4-NVLINK-INTRA"]="250"
    ["H100-SXM5-NVLINK-INTRA"]="350"
    ["V100-SXM2-NVLINK-INTRA"]="180"

    # Inter-node with InfiniBand
    ["A100-IB-HDR"]="180"      # 200 Gbps IB HDR
    ["A100-IB-NDR"]="360"      # 400 Gbps IB NDR
    ["H100-IB-NDR"]="360"      # 400 Gbps IB NDR
    ["H100-IB-XDR"]="720"      # 800 Gbps IB XDR (future)

    # Inter-node with RoCE
    ["A100-ROCE-100G"]="90"
    ["A100-ROCE-200G"]="180"
    ["H100-ROCE-400G"]="360"
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
            -n|--nodes)
                NODES_FILE="$2"
                shift 2
                ;;
            -g|--gpus-per-node)
                GPUS_PER_NODE="$2"
                shift 2
                ;;
            -i|--iterations)
                ITERATIONS="$2"
                shift 2
                ;;
            -s|--size)
                MESSAGE_SIZE="$2"
                shift 2
                ;;
            -o|--output)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            -t|--threshold)
                THRESHOLD="$2"
                shift 2
                ;;
            -b|--baseline)
                BASELINE_FILE="$2"
                shift 2
                ;;
            --mpi-path)
                MPI_PATH="$2"
                shift 2
                ;;
            --nccl-tests-path)
                NCCL_TESTS_PATH="$2"
                shift 2
                ;;
            --pairwise)
                PAIRWISE=1
                shift
                ;;
            --binary-search)
                BINARY_SEARCH=1
                shift
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

    # Validate required arguments
    if [ -z "$NODES_FILE" ]; then
        print_color "$RED" "ERROR: Nodes file is required (-n/--nodes)"
        show_usage
    fi

    if [ ! -f "$NODES_FILE" ]; then
        print_color "$RED" "ERROR: Nodes file not found: $NODES_FILE"
        exit 1
    fi
}

# Function to check dependencies
check_dependencies() {
    print_color "$BLUE" "=== Checking Dependencies ==="

    # Check for MPI
    if [ -z "$MPI_PATH" ]; then
        if command -v mpirun &> /dev/null; then
            MPI_PATH=$(dirname $(dirname $(which mpirun)))
            verbose "Auto-detected MPI at: $MPI_PATH"
        else
            print_color "$RED" "ERROR: MPI not found. Please install OpenMPI or MPICH"
            exit 1
        fi
    fi

    # Verify mpirun is accessible
    if ! command -v mpirun &> /dev/null; then
        print_color "$RED" "ERROR: mpirun not found in PATH"
        exit 1
    fi

    # Check for nccl-tests
    if [ -z "$NCCL_TESTS_PATH" ]; then
        # Try common installation paths
        local test_paths=(
            "/usr/local/nccl-tests/build"
            "$HOME/nccl-tests/build"
            "/opt/nccl-tests/build"
            "./nccl-tests/build"
        )

        for path in "${test_paths[@]}"; do
            if [ -f "$path/all_reduce_perf" ]; then
                NCCL_TESTS_PATH="$path"
                verbose "Auto-detected nccl-tests at: $NCCL_TESTS_PATH"
                break
            fi
        done
    fi

    if [ -z "$NCCL_TESTS_PATH" ] || [ ! -f "$NCCL_TESTS_PATH/all_reduce_perf" ]; then
        print_color "$RED" "ERROR: nccl-tests not found. Please install from https://github.com/NVIDIA/nccl-tests"
        print_color "$YELLOW" "Build instructions:"
        print_color "$YELLOW" "  git clone https://github.com/NVIDIA/nccl-tests.git"
        print_color "$YELLOW" "  cd nccl-tests"
        print_color "$YELLOW" "  make MPI=1 MPI_HOME=$MPI_PATH"
        exit 1
    fi

    ALL_REDUCE_PERF="$NCCL_TESTS_PATH/all_reduce_perf"

    print_color "$GREEN" "✓ Dependency check passed"
    print_color "$BLUE" "  MPI: $MPI_PATH"
    print_color "$BLUE" "  NCCL Tests: $NCCL_TESTS_PATH"
    echo
}

# Function to validate node connectivity
validate_nodes() {
    print_color "$BLUE" "=== Validating Node Connectivity ==="

    # Read nodes from file
    mapfile -t NODES < "$NODES_FILE"
    NODE_COUNT=${#NODES[@]}

    echo "Node count: $NODE_COUNT"
    echo "Nodes:"
    printf '  %s\n' "${NODES[@]}"

    # Test SSH connectivity
    local failed_nodes=0
    for node in "${NODES[@]}"; do
        if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "$node" "echo 'OK'" &>/dev/null; then
            print_color "$RED" "⚠ Cannot SSH to node: $node"
            failed_nodes=$((failed_nodes + 1))
        else
            verbose "✓ SSH connection to $node: OK"
        fi
    done

    if [ $failed_nodes -gt 0 ]; then
        print_color "$RED" "ERROR: Failed to connect to $failed_nodes node(s)"
        print_color "$YELLOW" "Please ensure SSH key-based authentication is configured"
        exit 1
    fi

    print_color "$GREEN" "✓ All nodes are accessible"

    # Auto-detect GPUs per node if not specified
    if [ $GPUS_PER_NODE -eq 0 ]; then
        local first_node="${NODES[0]}"
        GPUS_PER_NODE=$(ssh "$first_node" "nvidia-smi --query-gpu=count --format=csv,noheader | head -1")
        verbose "Auto-detected GPUs per node: $GPUS_PER_NODE"
    fi

    TOTAL_GPUS=$((NODE_COUNT * GPUS_PER_NODE))
    echo "GPUs per node: $GPUS_PER_NODE"
    echo "Total GPUs: $TOTAL_GPUS"

    mkdir -p "$OUTPUT_DIR"

    echo
}

# Function to run NCCL all-reduce test
run_nccl_test() {
    local nodes=$1
    local output_file=$2
    local iteration=${3:-1}

    verbose "Running NCCL test iteration $iteration on nodes: $nodes"

    # Create hostfile for MPI
    local hostfile="$OUTPUT_DIR/hostfile_${TIMESTAMP}_$$"
    echo "$nodes" | tr ',' '\n' | while read node; do
        echo "$node slots=$GPUS_PER_NODE"
    done > "$hostfile"

    # Count total processes
    local node_count=$(echo "$nodes" | tr ',' '\n' | wc -l)
    local nprocs=$((node_count * GPUS_PER_NODE))

    # Run all_reduce_perf
    local cmd="mpirun -np $nprocs \
        --hostfile $hostfile \
        --map-by ppr:${GPUS_PER_NODE}:node \
        -x NCCL_DEBUG=${NCCL_DEBUG:-WARN} \
        -x NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0} \
        $ALL_REDUCE_PERF \
        -b 8 \
        -e $MESSAGE_SIZE \
        -f 2 \
        -g 1"

    verbose "Command: $cmd"

    # Run the test
    if eval "$cmd" >> "$output_file" 2>&1; then
        rm -f "$hostfile"
        return 0
    else
        rm -f "$hostfile"
        return 1
    fi
}

# Function to parse NCCL test output and extract bus bandwidth
parse_nccl_output() {
    local output_file=$1

    # Extract the line with the largest message size (last line of results)
    # Example line: "   8589934592    11264    3823.7    2.25    3823.5    2.25     250.2"
    # Columns: size | count | time (us) | algbw (GB/s) | time (us) | algbw (GB/s) | busbw (GB/s)

    local busbw=$(grep -E "^\s+[0-9]+" "$output_file" | tail -1 | awk '{print $NF}')

    if [ -n "$busbw" ]; then
        echo "$busbw"
    else
        echo "0"
    fi
}

# Function to run multiple iterations and collect statistics
run_multiple_iterations() {
    local nodes=$1
    local test_name=$2

    print_color "$BLUE" "=== Running $ITERATIONS iterations for: $test_name ==="

    local output_base="$OUTPUT_DIR/${test_name}_${TIMESTAMP}"
    local bandwidths=()

    for i in $(seq 1 $ITERATIONS); do
        local iter_output="${output_base}_iter${i}.txt"

        echo -n "  Iteration $i/$ITERATIONS ... "

        if run_nccl_test "$nodes" "$iter_output" "$i"; then
            local bw=$(parse_nccl_output "$iter_output")

            if [ "$bw" != "0" ]; then
                bandwidths+=("$bw")
                echo "Bus BW: $bw GB/s"
            else
                print_color "$YELLOW" "FAILED (could not parse output)"
            fi
        else
            print_color "$RED" "FAILED (test error)"
        fi
    done

    # Calculate statistics
    if [ ${#bandwidths[@]} -eq 0 ]; then
        print_color "$RED" "ERROR: All iterations failed for $test_name"
        return 1
    fi

    local sum=0
    local min=${bandwidths[0]}
    local max=${bandwidths[0]}

    for bw in "${bandwidths[@]}"; do
        sum=$(echo "$sum + $bw" | bc -l)
        if (( $(echo "$bw < $min" | bc -l) )); then
            min=$bw
        fi
        if (( $(echo "$bw > $max" | bc -l) )); then
            max=$bw
        fi
    done

    local mean=$(echo "scale=2; $sum / ${#bandwidths[@]}" | bc -l)

    # Calculate standard deviation
    local sq_diff_sum=0
    for bw in "${bandwidths[@]}"; do
        local diff=$(echo "$bw - $mean" | bc -l)
        local sq_diff=$(echo "$diff * $diff" | bc -l)
        sq_diff_sum=$(echo "$sq_diff_sum + $sq_diff" | bc -l)
    done
    local variance=$(echo "scale=2; $sq_diff_sum / ${#bandwidths[@]}" | bc -l)
    local stddev=$(echo "scale=2; sqrt($variance)" | bc -l)

    # Save statistics
    local stats_file="${output_base}_stats.txt"
    cat > "$stats_file" <<EOF
Test: $test_name
Nodes: $nodes
Iterations: ${#bandwidths[@]}
Mean: $mean GB/s
StdDev: $stddev GB/s
Min: $min GB/s
Max: $max GB/s
All values: ${bandwidths[*]}
EOF

    echo ""
    print_color "$GREEN" "Statistics:"
    cat "$stats_file"
    echo ""

    # Return mean bandwidth
    echo "$mean"
}

# Function to test all nodes together
test_all_nodes() {
    print_color "$BLUE" "=== Testing All Nodes Together ==="

    local all_nodes=$(IFS=,; echo "${NODES[*]}")
    local mean_bw=$(run_multiple_iterations "$all_nodes" "all_nodes")

    if [ -z "$mean_bw" ] || [ "$mean_bw" = "0" ]; then
        print_color "$RED" "ERROR: Could not determine baseline performance"
        exit 1
    fi

    ALL_NODES_BANDWIDTH=$mean_bw

    echo ""
}

# Function to perform pairwise node testing
test_pairwise_nodes() {
    if [ $PAIRWISE -eq 0 ]; then
        verbose "Pairwise testing disabled"
        return
    fi

    print_color "$BLUE" "=== Performing Pairwise Node Testing ==="

    local pairwise_results="$OUTPUT_DIR/pairwise_results_${TIMESTAMP}.csv"
    echo "Node1,Node2,Mean_BW_GB/s,StdDev,Status" > "$pairwise_results"

    local slow_pairs=()

    for i in "${!NODES[@]}"; do
        for j in "${!NODES[@]}"; do
            if [ $i -ge $j ]; then continue; fi  # Skip duplicates and self-pairs

            local node1="${NODES[$i]}"
            local node2="${NODES[$j]}"
            local pair_name="node${i}_node${j}"

            echo ""
            print_color "$BLUE" "Testing pair: $node1 <-> $node2"

            local mean_bw=$(run_multiple_iterations "$node1,$node2" "$pair_name")

            # Check if this pair is significantly slower
            local threshold_bw=$(echo "$ALL_NODES_BANDWIDTH * $THRESHOLD / 100" | bc -l)
            local is_slow=$(echo "$mean_bw < $threshold_bw" | bc -l)

            local status="OK"
            if [ "$is_slow" -eq 1 ]; then
                status="SLOW"
                slow_pairs+=("$node1,$node2")
                print_color "$RED" "⚠ Slow pair detected: $node1 <-> $node2 (${mean_bw} GB/s)"
            fi

            # Read stats file to get stddev
            local stats_file="$OUTPUT_DIR/${pair_name}_${TIMESTAMP}_stats.txt"
            local stddev=$(grep "StdDev:" "$stats_file" | awk '{print $2}')

            echo "$node1,$node2,$mean_bw,$stddev,$status" >> "$pairwise_results"
        done
    done

    echo ""
    print_color "$BLUE" "=== Pairwise Test Results ==="
    column -t -s',' "$pairwise_results"

    if [ ${#slow_pairs[@]} -gt 0 ]; then
        echo ""
        print_color "$RED" "⚠ Slow pairs detected:"
        printf '  %s\n' "${slow_pairs[@]}"

        # Analyze which nodes appear most frequently in slow pairs
        analyze_slow_pairs "${slow_pairs[@]}"
    else
        print_color "$GREEN" "✓ All node pairs perform within expected range"
    fi

    echo ""
}

# Function to analyze slow pairs and identify problematic nodes
analyze_slow_pairs() {
    local pairs=("$@")

    print_color "$BLUE" "=== Analyzing Slow Pairs ==="

    declare -A node_count

    for pair in "${pairs[@]}"; do
        local node1=$(echo "$pair" | cut -d',' -f1)
        local node2=$(echo "$pair" | cut -d',' -f2)

        node_count[$node1]=$((${node_count[$node1]:-0} + 1))
        node_count[$node2]=$((${node_count[$node2]:-0} + 1))
    done

    echo "Node appearance in slow pairs:"
    for node in "${!node_count[@]}"; do
        local count=${node_count[$node]}
        if [ $count -gt 1 ]; then
            print_color "$RED" "  $node: $count times (LIKELY PROBLEMATIC)"
        else
            echo "  $node: $count time"
        fi
    done

    echo ""
}

# Function to perform binary search for slow nodes
binary_search_slow_nodes() {
    if [ $BINARY_SEARCH -eq 0 ]; then
        verbose "Binary search disabled"
        return
    fi

    if [ $NODE_COUNT -lt 4 ]; then
        print_color "$YELLOW" "Binary search requires at least 4 nodes, skipping"
        return
    fi

    print_color "$BLUE" "=== Performing Binary Search for Slow Nodes ==="

    # Recursively test node subsets
    binary_search_recursive 0 $((NODE_COUNT - 1)) 1
}

# Recursive binary search function
binary_search_recursive() {
    local start=$1
    local end=$2
    local depth=$3

    local count=$((end - start + 1))

    if [ $count -lt 2 ]; then
        return
    fi

    echo ""
    print_color "$BLUE" "Binary search depth $depth: testing nodes $start to $end ($count nodes)"

    # Get subset of nodes
    local subset_nodes=""
    for i in $(seq $start $end); do
        if [ -n "$subset_nodes" ]; then
            subset_nodes="${subset_nodes},${NODES[$i]}"
        else
            subset_nodes="${NODES[$i]}"
        fi
    done

    local test_name="binary_search_d${depth}_n${start}-${end}"
    local mean_bw=$(run_multiple_iterations "$subset_nodes" "$test_name")

    # Check if this subset is slow
    local threshold_bw=$(echo "$ALL_NODES_BANDWIDTH * $THRESHOLD / 100" | bc -l)
    local is_slow=$(echo "$mean_bw < $threshold_bw" | bc -l)

    if [ "$is_slow" -eq 1 ]; then
        print_color "$YELLOW" "⚠ Slow performance detected in this subset (${mean_bw} GB/s)"

        if [ $count -eq 2 ]; then
            print_color "$RED" "⚠ Problematic nodes identified: ${NODES[$start]}, ${NODES[$end]}"
        else
            # Split and continue searching
            local mid=$(( (start + end) / 2 ))

            echo "Splitting into two subsets..."
            binary_search_recursive $start $mid $((depth + 1))
            binary_search_recursive $((mid + 1)) $end $((depth + 1))
        fi
    else
        print_color "$GREEN" "✓ This subset performs normally (${mean_bw} GB/s)"
    fi
}

# Function to generate comprehensive report
generate_report() {
    print_color "$BLUE" "=== Generating Comprehensive Report ==="

    local report_file="$OUTPUT_DIR/nccl_check_report_${TIMESTAMP}.md"

    cat > "$report_file" <<EOF
# Inter-Node NCCL Communication Check Report

**Timestamp:** $(date)
**Node Count:** $NODE_COUNT
**GPUs per Node:** $GPUS_PER_NODE
**Total GPUs:** $TOTAL_GPUS
**Iterations:** $ITERATIONS
**Message Size:** $MESSAGE_SIZE
**Threshold:** ${THRESHOLD}%

---

## Cluster Configuration

**Nodes:**
EOF

    printf '- %s\n' "${NODES[@]}" >> "$report_file"

    cat >> "$report_file" <<EOF

---

## All-Nodes Test Results

**Mean Bus Bandwidth:** $ALL_NODES_BANDWIDTH GB/s

EOF

    # Add all-nodes statistics
    if [ -f "$OUTPUT_DIR/all_nodes_${TIMESTAMP}_stats.txt" ]; then
        echo '```' >> "$report_file"
        cat "$OUTPUT_DIR/all_nodes_${TIMESTAMP}_stats.txt" >> "$report_file"
        echo '```' >> "$report_file"
    fi

    # Add pairwise results if available
    if [ -f "$OUTPUT_DIR/pairwise_results_${TIMESTAMP}.csv" ]; then
        echo "" >> "$report_file"
        echo "## Pairwise Test Results" >> "$report_file"
        echo "" >> "$report_file"
        echo '```' >> "$report_file"
        column -t -s',' "$OUTPUT_DIR/pairwise_results_${TIMESTAMP}.csv" >> "$report_file"
        echo '```' >> "$report_file"
    fi

    # Add recommendations
    echo "" >> "$report_file"
    echo "## Recommendations" >> "$report_file"
    echo "" >> "$report_file"

    # Check if any slow nodes were detected
    if [ -f "$OUTPUT_DIR/pairwise_results_${TIMESTAMP}.csv" ]; then
        local slow_count=$(grep -c "SLOW" "$OUTPUT_DIR/pairwise_results_${TIMESTAMP}.csv" || true)

        if [ "$slow_count" -gt 0 ]; then
            echo "- $slow_count slow node pair(s) detected" >> "$report_file"
            echo "- Investigate nodes that appear frequently in slow pairs" >> "$report_file"
            echo "- Check InfiniBand/network connectivity" >> "$report_file"
            echo "- Verify NCCL and network driver versions" >> "$report_file"
            echo "- Check for network congestion or misconfigurations" >> "$report_file"
        else
            echo "- All node pairs perform within expected range" >> "$report_file"
            echo "- System is performing normally" >> "$report_file"
        fi
    fi

    echo "" >> "$report_file"
    echo "---" >> "$report_file"
    echo "Report generated by: inter_node_nccl_check.sh" >> "$report_file"

    print_color "$GREEN" "✓ Report saved to: $report_file"
    echo
}

# Main execution
main() {
    parse_args "$@"

    print_color "$GREEN" "========================================"
    print_color "$GREEN" "   Inter-Node NCCL Communication Check"
    print_color "$GREEN" "========================================"
    echo

    check_dependencies
    validate_nodes
    test_all_nodes
    test_pairwise_nodes
    binary_search_slow_nodes
    generate_report

    print_color "$GREEN" "========================================"
    print_color "$GREEN" "   NCCL Check Complete"
    print_color "$GREEN" "========================================"
    print_color "$BLUE" "Results saved to: $OUTPUT_DIR"
    echo
}

# Run main function
main "$@"

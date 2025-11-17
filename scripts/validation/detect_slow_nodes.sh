#!/bin/bash
################################################################################
# Comprehensive Slow Node Detection Tool
#
# Purpose: Detect slow nodes and GPUs in a cluster by running:
#   1. Intra-node bandwidth checks (NVLink, PCIe) on each node
#   2. Inter-node NCCL communication tests
#   3. Statistical analysis to identify outliers
#   4. Comprehensive reporting with recommendations
#
# Usage: ./detect_slow_nodes.sh [options]
#
# Options:
#   -n, --nodes FILE          File containing list of nodes (one per line)
#   -o, --output DIR          Output directory for results
#   -t, --threshold PCT       Performance threshold percentage (default: 90)
#   --skip-intra              Skip intra-node bandwidth checks
#   --skip-inter              Skip inter-node NCCL checks
#   --pairwise                Enable pairwise node testing
#   --binary-search           Enable binary search for slow node detection
#   --nccl-iterations N       Number of NCCL test iterations (default: 10)
#   --parallel                Run intra-node checks in parallel
#   -v, --verbose             Verbose output
#   -h, --help                Show this help message
#
# Examples:
#   # Full cluster check
#   ./detect_slow_nodes.sh -n nodes.txt -o results
#
#   # Only intra-node checks
#   ./detect_slow_nodes.sh -n nodes.txt --skip-inter
#
#   # Only inter-node checks with pairwise testing
#   ./detect_slow_nodes.sh -n nodes.txt --skip-intra --pairwise
#
#   # Parallel intra-node checks with binary search
#   ./detect_slow_nodes.sh -n nodes.txt --parallel --binary-search
#
################################################################################

set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default configuration
NODES_FILE=""
OUTPUT_DIR="./slow_node_detection_results"
THRESHOLD=90
SKIP_INTRA=0
SKIP_INTER=0
PAIRWISE=0
BINARY_SEARCH=0
NCCL_ITERATIONS=10
PARALLEL=0
VERBOSE=0
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Function to print colored output
print_color() {
    local color=$1
    shift
    echo -e "${color}$@${NC}"
}

# Function to print section header
print_header() {
    echo ""
    print_color "$CYAN" "========================================"
    print_color "$CYAN" "  $1"
    print_color "$CYAN" "========================================"
    echo ""
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
            -o|--output)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            -t|--threshold)
                THRESHOLD="$2"
                shift 2
                ;;
            --skip-intra)
                SKIP_INTRA=1
                shift
                ;;
            --skip-inter)
                SKIP_INTER=1
                shift
                ;;
            --pairwise)
                PAIRWISE=1
                shift
                ;;
            --binary-search)
                BINARY_SEARCH=1
                shift
                ;;
            --nccl-iterations)
                NCCL_ITERATIONS="$2"
                shift 2
                ;;
            --parallel)
                PARALLEL=1
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

    if [ $SKIP_INTRA -eq 1 ] && [ $SKIP_INTER -eq 1 ]; then
        print_color "$RED" "ERROR: Cannot skip both intra-node and inter-node checks"
        exit 1
    fi
}

# Function to check script dependencies
check_script_dependencies() {
    print_header "Checking Script Dependencies"

    local missing_scripts=0

    INTRA_NODE_SCRIPT="$SCRIPT_DIR/intra_node_bandwidth_check.sh"
    INTER_NODE_SCRIPT="$SCRIPT_DIR/inter_node_nccl_check.sh"

    if [ $SKIP_INTRA -eq 0 ]; then
        if [ ! -f "$INTRA_NODE_SCRIPT" ]; then
            print_color "$RED" "ERROR: Intra-node script not found: $INTRA_NODE_SCRIPT"
            missing_scripts=1
        else
            chmod +x "$INTRA_NODE_SCRIPT"
            verbose "Found intra-node script: $INTRA_NODE_SCRIPT"
        fi
    fi

    if [ $SKIP_INTER -eq 0 ]; then
        if [ ! -f "$INTER_NODE_SCRIPT" ]; then
            print_color "$RED" "ERROR: Inter-node script not found: $INTER_NODE_SCRIPT"
            missing_scripts=1
        else
            chmod +x "$INTER_NODE_SCRIPT"
            verbose "Found inter-node script: $INTER_NODE_SCRIPT"
        fi
    fi

    if [ $missing_scripts -eq 1 ]; then
        exit 1
    fi

    print_color "$GREEN" "✓ All required scripts found"
}

# Function to validate nodes
validate_nodes() {
    print_header "Validating Cluster Nodes"

    # Read nodes from file
    mapfile -t NODES < "$NODES_FILE"
    NODE_COUNT=${#NODES[@]}

    echo "Node count: $NODE_COUNT"
    echo "Nodes:"
    printf '  %s\n' "${NODES[@]}"

    # Create output directory
    mkdir -p "$OUTPUT_DIR"

    # Test SSH connectivity
    local failed_nodes=()
    for node in "${NODES[@]}"; do
        echo -n "Testing connectivity to $node ... "
        if ssh -o ConnectTimeout=5 -o BatchMode=yes "$node" "echo 'OK'" &>/dev/null; then
            print_color "$GREEN" "✓"
        else
            print_color "$RED" "✗ FAILED"
            failed_nodes+=("$node")
        fi
    done

    if [ ${#failed_nodes[@]} -gt 0 ]; then
        print_color "$RED" "ERROR: Cannot connect to ${#failed_nodes[@]} node(s):"
        printf '  %s\n' "${failed_nodes[@]}"
        exit 1
    fi

    print_color "$GREEN" "✓ All nodes are accessible"
    echo ""
}

# Function to run intra-node bandwidth checks
run_intra_node_checks() {
    if [ $SKIP_INTRA -eq 1 ]; then
        verbose "Skipping intra-node bandwidth checks"
        return
    fi

    print_header "Running Intra-Node Bandwidth Checks"

    local intra_output_dir="$OUTPUT_DIR/intra_node_results"
    mkdir -p "$intra_output_dir"

    local failed_nodes=()
    local slow_gpus_detected=0

    if [ $PARALLEL -eq 1 ]; then
        print_color "$BLUE" "Running checks in parallel mode..."

        # Array to store background job PIDs
        local pids=()

        for node in "${NODES[@]}"; do
            local node_output="$intra_output_dir/${node}_${TIMESTAMP}"
            mkdir -p "$node_output"

            verbose "Starting intra-node check on $node (background)"

            # Run remotely in background
            (
                ssh "$node" "bash -s" -- < "$INTRA_NODE_SCRIPT" \
                    -o "$node_output" \
                    -t "$THRESHOLD" \
                    $([ $VERBOSE -eq 1 ] && echo "-v") \
                    > "$node_output/console_output.log" 2>&1

                echo $? > "$node_output/exit_code"
            ) &

            pids+=($!)
        done

        # Wait for all background jobs
        local job_num=0
        for pid in "${pids[@]}"; do
            local node="${NODES[$job_num]}"
            echo -n "Waiting for $node ... "

            if wait $pid; then
                print_color "$GREEN" "✓ Completed"
            else
                print_color "$RED" "✗ Failed"
                failed_nodes+=("$node")
            fi

            job_num=$((job_num + 1))
        done

    else
        print_color "$BLUE" "Running checks in sequential mode..."

        for node in "${NODES[@]}"; do
            local node_output="$intra_output_dir/${node}_${TIMESTAMP}"
            mkdir -p "$node_output"

            echo ""
            print_color "$BLUE" "Checking node: $node"

            # Run remotely
            if ssh "$node" "bash -s" -- < "$INTRA_NODE_SCRIPT" \
                -o "$node_output" \
                -t "$THRESHOLD" \
                $([ $VERBOSE -eq 1 ] && echo "-v"); then

                print_color "$GREEN" "✓ Intra-node check completed for $node"
            else
                print_color "$RED" "✗ Intra-node check failed for $node"
                failed_nodes+=("$node")
            fi
        done
    fi

    echo ""

    # Analyze results
    print_color "$BLUE" "=== Analyzing Intra-Node Results ==="

    for node in "${NODES[@]}"; do
        local node_output="$intra_output_dir/${node}_${TIMESTAMP}"

        # Check for slow connections file
        local slow_file=$(find "$node_output" -name "slow_connections_*.txt" 2>/dev/null | head -1)

        if [ -n "$slow_file" ] && [ -s "$slow_file" ]; then
            print_color "$YELLOW" "⚠ Slow GPU connections detected on $node:"
            cat "$slow_file" | sed 's/^/    /'
            slow_gpus_detected=1
        fi
    done

    if [ $slow_gpus_detected -eq 0 ]; then
        print_color "$GREEN" "✓ No slow intra-node connections detected"
    fi

    if [ ${#failed_nodes[@]} -gt 0 ]; then
        print_color "$YELLOW" "⚠ Intra-node checks failed on ${#failed_nodes[@]} node(s):"
        printf '  %s\n' "${failed_nodes[@]}"
    fi

    echo ""
}

# Function to run inter-node NCCL checks
run_inter_node_checks() {
    if [ $SKIP_INTER -eq 1 ]; then
        verbose "Skipping inter-node NCCL checks"
        return
    fi

    print_header "Running Inter-Node NCCL Checks"

    local inter_output_dir="$OUTPUT_DIR/inter_node_results"
    mkdir -p "$inter_output_dir"

    local inter_args="-n $NODES_FILE -o $inter_output_dir -t $THRESHOLD -i $NCCL_ITERATIONS"

    if [ $PAIRWISE -eq 1 ]; then
        inter_args="$inter_args --pairwise"
    fi

    if [ $BINARY_SEARCH -eq 1 ]; then
        inter_args="$inter_args --binary-search"
    fi

    if [ $VERBOSE -eq 1 ]; then
        inter_args="$inter_args -v"
    fi

    verbose "Running: $INTER_NODE_SCRIPT $inter_args"

    if bash "$INTER_NODE_SCRIPT" $inter_args; then
        print_color "$GREEN" "✓ Inter-node NCCL checks completed"
    else
        print_color "$RED" "✗ Inter-node NCCL checks failed"
    fi

    echo ""
}

# Function to aggregate and analyze all results
aggregate_results() {
    print_header "Aggregating Results"

    local summary_file="$OUTPUT_DIR/slow_node_summary_${TIMESTAMP}.md"

    cat > "$summary_file" <<EOF
# Slow Node Detection Summary

**Timestamp:** $(date)
**Node Count:** $NODE_COUNT
**Threshold:** ${THRESHOLD}%

---

## Cluster Configuration

**Nodes:**
EOF

    printf '- %s\n' "${NODES[@]}" >> "$summary_file"

    echo "" >> "$summary_file"
    echo "---" >> "$summary_file"
    echo "" >> "$summary_file"

    # Add intra-node results summary
    if [ $SKIP_INTRA -eq 0 ]; then
        echo "## Intra-Node Bandwidth Check Results" >> "$summary_file"
        echo "" >> "$summary_file"

        local intra_issues=0

        for node in "${NODES[@]}"; do
            local node_output="$OUTPUT_DIR/intra_node_results/${node}_${TIMESTAMP}"

            echo "### Node: $node" >> "$summary_file"
            echo "" >> "$summary_file"

            # Check if report exists
            local report=$(find "$node_output" -name "bandwidth_check_report_*.md" 2>/dev/null | head -1)

            if [ -n "$report" ] && [ -f "$report" ]; then
                # Extract key information
                echo '```' >> "$summary_file"
                grep -A 10 "## Summary" "$report" | tail -9 >> "$summary_file" 2>/dev/null || echo "Report found but could not extract summary" >> "$summary_file"
                echo '```' >> "$summary_file"

                # Check for issues
                if grep -q "Issues Detected" "$report"; then
                    print_color "$YELLOW" "⚠ Issues detected on $node"
                    intra_issues=$((intra_issues + 1))

                    echo "" >> "$summary_file"
                    echo "**⚠ Issues Detected:**" >> "$summary_file"
                    echo '```' >> "$summary_file"
                    sed -n '/### ⚠ Issues Detected/,/###/p' "$report" | head -n -1 | tail -n +2 >> "$summary_file"
                    echo '```' >> "$summary_file"
                fi
            else
                echo "*No report generated*" >> "$summary_file"
            fi

            echo "" >> "$summary_file"
        done

        if [ $intra_issues -eq 0 ]; then
            print_color "$GREEN" "✓ No intra-node issues detected"
        else
            print_color "$YELLOW" "⚠ Intra-node issues detected on $intra_issues node(s)"
        fi

        echo "" >> "$summary_file"
        echo "---" >> "$summary_file"
        echo "" >> "$summary_file"
    fi

    # Add inter-node results summary
    if [ $SKIP_INTER -eq 0 ]; then
        echo "## Inter-Node NCCL Check Results" >> "$summary_file"
        echo "" >> "$summary_file"

        local inter_report=$(find "$OUTPUT_DIR/inter_node_results" -name "nccl_check_report_*.md" 2>/dev/null | head -1)

        if [ -n "$inter_report" ] && [ -f "$inter_report" ]; then
            # Extract key sections
            sed -n '/## All-Nodes Test Results/,/## Recommendations/p' "$inter_report" | head -n -1 >> "$summary_file"

            # Check for slow pairs
            if grep -q "SLOW" "$OUTPUT_DIR/inter_node_results/pairwise_results_"*.csv 2>/dev/null; then
                print_color "$YELLOW" "⚠ Slow node pairs detected"

                echo "" >> "$summary_file"
                echo "### ⚠ Slow Node Pairs Detected" >> "$summary_file"
                echo "" >> "$summary_file"
                echo '```' >> "$summary_file"
                grep "SLOW" "$OUTPUT_DIR/inter_node_results/pairwise_results_"*.csv | head -20 >> "$summary_file"
                echo '```' >> "$summary_file"
            else
                print_color "$GREEN" "✓ No slow node pairs detected"
            fi
        else
            echo "*No inter-node report generated*" >> "$summary_file"
        fi

        echo "" >> "$summary_file"
    fi

    # Add overall recommendations
    echo "---" >> "$summary_file"
    echo "" >> "$summary_file"
    echo "## Overall Recommendations" >> "$summary_file"
    echo "" >> "$summary_file"

    # Determine if any issues were found
    local total_issues=0

    if [ -f "$OUTPUT_DIR/intra_node_results/"*"/slow_connections_"*.txt 2>/dev/null ]; then
        total_issues=$((total_issues + 1))
        echo "- **Intra-node bandwidth issues detected**: Check NVLink cables and PCIe connections" >> "$summary_file"
    fi

    if grep -q "SLOW" "$OUTPUT_DIR/inter_node_results/pairwise_results_"*.csv 2>/dev/null; then
        total_issues=$((total_issues + 1))
        echo "- **Inter-node communication issues detected**: Check network fabric (InfiniBand/RoCE)" >> "$summary_file"
        echo "- Investigate nodes that appear frequently in slow pairs" >> "$summary_file"
    fi

    if [ $total_issues -eq 0 ]; then
        echo "- ✓ All tests passed successfully" >> "$summary_file"
        echo "- ✓ No performance issues detected" >> "$summary_file"
        echo "- ✓ Cluster is ready for production workloads" >> "$summary_file"
    else
        echo "- Review detailed reports for each affected node" >> "$summary_file"
        echo "- Re-run tests after addressing issues" >> "$summary_file"
    fi

    echo "" >> "$summary_file"
    echo "---" >> "$summary_file"
    echo "" >> "$summary_file"
    echo "**Detailed Results:**" >> "$summary_file"
    echo "- Intra-node results: \`$OUTPUT_DIR/intra_node_results/\`" >> "$summary_file"
    echo "- Inter-node results: \`$OUTPUT_DIR/inter_node_results/\`" >> "$summary_file"
    echo "" >> "$summary_file"

    print_color "$GREEN" "✓ Summary report generated: $summary_file"
    echo ""

    # Display summary
    print_header "Summary"
    cat "$summary_file"
}

# Main execution
main() {
    parse_args "$@"

    print_header "Comprehensive Slow Node Detection"

    echo "Configuration:"
    echo "  Nodes file: $NODES_FILE"
    echo "  Output directory: $OUTPUT_DIR"
    echo "  Threshold: ${THRESHOLD}%"
    echo "  Skip intra-node checks: $([ $SKIP_INTRA -eq 1 ] && echo 'Yes' || echo 'No')"
    echo "  Skip inter-node checks: $([ $SKIP_INTER -eq 1 ] && echo 'Yes' || echo 'No')"
    echo "  Pairwise testing: $([ $PAIRWISE -eq 1 ] && echo 'Enabled' || echo 'Disabled')"
    echo "  Binary search: $([ $BINARY_SEARCH -eq 1 ] && echo 'Enabled' || echo 'Disabled')"
    echo "  Parallel mode: $([ $PARALLEL -eq 1 ] && echo 'Enabled' || echo 'Disabled')"
    echo "  NCCL iterations: $NCCL_ITERATIONS"
    echo ""

    check_script_dependencies
    validate_nodes
    run_intra_node_checks
    run_inter_node_checks
    aggregate_results

    print_header "Detection Complete"

    print_color "$GREEN" "All results saved to: $OUTPUT_DIR"
    print_color "$CYAN" "Summary report: $OUTPUT_DIR/slow_node_summary_${TIMESTAMP}.md"
    echo ""
}

# Run main function
main "$@"

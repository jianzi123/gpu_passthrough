#!/bin/bash
# NCCL Benchmark Testing Script
# Tests allreduce, broadcast, and other collective operations
# Compares against performance baselines

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
OUTPUT_DIR="${1:-/tmp/nccl_benchmarks}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NCCL_TESTS_DIR="${NCCL_TESTS_DIR:-/opt/nccl-tests/build}"

# Default test parameters
MIN_SIZE="${MIN_SIZE:-8}"          # 8 bytes
MAX_SIZE="${MAX_SIZE:-8G}"         # 8 GB
STEP_FACTOR="${STEP_FACTOR:-2}"    # 2x each step
NUM_WARMUP="${NUM_WARMUP:-5}"      # 5 warmup iterations
NUM_ITERS="${NUM_ITERS:-20}"       # 20 test iterations

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "NCCL Benchmark Suite"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Timestamp: $TIMESTAMP"
echo ""

# Check if NCCL tests are available
if [ ! -d "$NCCL_TESTS_DIR" ]; then
    echo -e "${RED}ERROR: NCCL tests not found at $NCCL_TESTS_DIR${NC}"
    echo ""
    echo "To install NCCL tests:"
    echo "  git clone https://github.com/NVIDIA/nccl-tests.git"
    echo "  cd nccl-tests"
    echo "  make MPI=1"
    echo ""
    echo "Then set NCCL_TESTS_DIR=/path/to/nccl-tests/build"
    exit 1
fi

# Detect system configuration
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -d ' ')

echo "System Configuration:"
echo "  GPU Model: $GPU_MODEL"
echo "  GPU Count: $GPU_COUNT"
echo "  NCCL Version: $(cat /usr/local/cuda/include/nccl.h 2>/dev/null | grep '#define NCCL_VERSION_CODE' | awk '{print $3}' || echo 'Unknown')"
echo ""

# Check if running with MPI
if command -v mpirun &> /dev/null; then
    MPI_AVAILABLE=true
    MPI_VERSION=$(mpirun --version | head -1)
    echo "  MPI: $MPI_VERSION"
else
    MPI_AVAILABLE=false
    echo "  MPI: Not available (single-node only)"
fi
echo ""

#==========================================
# Test 1: AllReduce Benchmark
#==========================================
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Test 1: AllReduce Benchmark${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

if [ "$MPI_AVAILABLE" = true ]; then
    # Multi-GPU test with MPI
    echo "Running AllReduce with $GPU_COUNT GPUs..."
    mpirun -np $GPU_COUNT \
        --bind-to none \
        --allow-run-as-root \
        "$NCCL_TESTS_DIR/all_reduce_perf" \
        -b $MIN_SIZE -e $MAX_SIZE -f $STEP_FACTOR \
        -g 1 -w $NUM_WARMUP -n $NUM_ITERS \
        | tee "$OUTPUT_DIR/allreduce_${GPU_COUNT}gpu_${TIMESTAMP}.txt"
else
    # Single-node multi-GPU test
    echo "Running AllReduce with $GPU_COUNT GPUs (single node)..."
    "$NCCL_TESTS_DIR/all_reduce_perf" \
        -b $MIN_SIZE -e $MAX_SIZE -f $STEP_FACTOR \
        -g $GPU_COUNT -w $NUM_WARMUP -n $NUM_ITERS \
        | tee "$OUTPUT_DIR/allreduce_${GPU_COUNT}gpu_${TIMESTAMP}.txt"
fi

# Parse peak bandwidth
ALLREDUCE_BUSBW=$(grep "float" "$OUTPUT_DIR/allreduce_${GPU_COUNT}gpu_${TIMESTAMP}.txt" | \
    awk '{print $12}' | sort -n | tail -1)

echo ""
echo "AllReduce Peak Bus Bandwidth: ${ALLREDUCE_BUSBW} GB/s"

#==========================================
# Test 2: Broadcast Benchmark
#==========================================
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Test 2: Broadcast Benchmark${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

if [ "$MPI_AVAILABLE" = true ]; then
    mpirun -np $GPU_COUNT \
        --bind-to none \
        --allow-run-as-root \
        "$NCCL_TESTS_DIR/broadcast_perf" \
        -b $MIN_SIZE -e $MAX_SIZE -f $STEP_FACTOR \
        -g 1 -w $NUM_WARMUP -n $NUM_ITERS \
        | tee "$OUTPUT_DIR/broadcast_${GPU_COUNT}gpu_${TIMESTAMP}.txt"
else
    "$NCCL_TESTS_DIR/broadcast_perf" \
        -b $MIN_SIZE -e $MAX_SIZE -f $STEP_FACTOR \
        -g $GPU_COUNT -w $NUM_WARMUP -n $NUM_ITERS \
        | tee "$OUTPUT_DIR/broadcast_${GPU_COUNT}gpu_${TIMESTAMP}.txt"
fi

BROADCAST_BUSBW=$(grep "float" "$OUTPUT_DIR/broadcast_${GPU_COUNT}gpu_${TIMESTAMP}.txt" | \
    awk '{print $12}' | sort -n | tail -1)

echo ""
echo "Broadcast Peak Bus Bandwidth: ${BROADCAST_BUSBW} GB/s"

#==========================================
# Test 3: Reduce Scatter Benchmark
#==========================================
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Test 3: Reduce-Scatter Benchmark${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

if [ "$MPI_AVAILABLE" = true ]; then
    mpirun -np $GPU_COUNT \
        --bind-to none \
        --allow-run-as-root \
        "$NCCL_TESTS_DIR/reduce_scatter_perf" \
        -b $MIN_SIZE -e $MAX_SIZE -f $STEP_FACTOR \
        -g 1 -w $NUM_WARMUP -n $NUM_ITERS \
        | tee "$OUTPUT_DIR/reduce_scatter_${GPU_COUNT}gpu_${TIMESTAMP}.txt"
else
    "$NCCL_TESTS_DIR/reduce_scatter_perf" \
        -b $MIN_SIZE -e $MAX_SIZE -f $STEP_FACTOR \
        -g $GPU_COUNT -w $NUM_WARMUP -n $NUM_ITERS \
        | tee "$OUTPUT_DIR/reduce_scatter_${GPU_COUNT}gpu_${TIMESTAMP}.txt"
fi

REDUCE_SCATTER_BUSBW=$(grep "float" "$OUTPUT_DIR/reduce_scatter_${GPU_COUNT}gpu_${TIMESTAMP}.txt" | \
    awk '{print $12}' | sort -n | tail -1)

echo ""
echo "Reduce-Scatter Peak Bus Bandwidth: ${REDUCE_SCATTER_BUSBW} GB/s"

#==========================================
# Test 4: All-Gather Benchmark
#==========================================
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Test 4: All-Gather Benchmark${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

if [ "$MPI_AVAILABLE" = true ]; then
    mpirun -np $GPU_COUNT \
        --bind-to none \
        --allow-run-as-root \
        "$NCCL_TESTS_DIR/allgather_perf" \
        -b $MIN_SIZE -e $MAX_SIZE -f $STEP_FACTOR \
        -g 1 -w $NUM_WARMUP -n $NUM_ITERS \
        | tee "$OUTPUT_DIR/allgather_${GPU_COUNT}gpu_${TIMESTAMP}.txt"
else
    "$NCCL_TESTS_DIR/allgather_perf" \
        -b $MIN_SIZE -e $MAX_SIZE -f $STEP_FACTOR \
        -g $GPU_COUNT -w $NUM_WARMUP -n $NUM_ITERS \
        | tee "$OUTPUT_DIR/allgather_${GPU_COUNT}gpu_${TIMESTAMP}.txt"
fi

ALLGATHER_BUSBW=$(grep "float" "$OUTPUT_DIR/allgather_${GPU_COUNT}gpu_${TIMESTAMP}.txt" | \
    awk '{print $12}' | sort -n | tail -1)

echo ""
echo "All-Gather Peak Bus Bandwidth: ${ALLGATHER_BUSBW} GB/s"

#==========================================
# Generate Summary Report
#==========================================
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}NCCL Benchmark Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Create JSON summary
cat > "$OUTPUT_DIR/nccl_summary_${TIMESTAMP}.json" << EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "gpu_model": "$GPU_MODEL",
  "gpu_count": $GPU_COUNT,
  "nccl_tests": {
    "allreduce_busbw_gbs": ${ALLREDUCE_BUSBW:-0},
    "broadcast_busbw_gbs": ${BROADCAST_BUSBW:-0},
    "reduce_scatter_busbw_gbs": ${REDUCE_SCATTER_BUSBW:-0},
    "allgather_busbw_gbs": ${ALLGATHER_BUSBW:-0}
  },
  "test_parameters": {
    "min_size": "$MIN_SIZE",
    "max_size": "$MAX_SIZE",
    "step_factor": $STEP_FACTOR,
    "warmup_iters": $NUM_WARMUP,
    "test_iters": $NUM_ITERS
  }
}
EOF

echo "Test Results:"
echo "  AllReduce Bus BW:      ${ALLREDUCE_BUSBW:-N/A} GB/s"
echo "  Broadcast Bus BW:      ${BROADCAST_BUSBW:-N/A} GB/s"
echo "  Reduce-Scatter Bus BW: ${REDUCE_SCATTER_BUSBW:-N/A} GB/s"
echo "  All-Gather Bus BW:     ${ALLGATHER_BUSBW:-N/A} GB/s"
echo ""

# Compare against baselines
BASELINE_SCRIPT="$(dirname $0)/../utils/performance_baselines.py"
if [ -f "$BASELINE_SCRIPT" ]; then
    echo "Expected Performance (from baselines):"
    python3 - <<PYTHON_SCRIPT
import sys
sys.path.insert(0, '$(dirname $BASELINE_SCRIPT)')
from performance_baselines import get_gpu_baseline

gpu_model = "$GPU_MODEL"
baseline = get_gpu_baseline(gpu_model)

if baseline:
    nccl_baseline = baseline.get('nccl_allreduce_busbw_gbs', {})
    intra = nccl_baseline.get('intra_node_${GPU_COUNT}gpu', 'N/A')
    print(f"  Expected AllReduce (intra-node): {intra} GB/s")

    # Performance check
    actual = ${ALLREDUCE_BUSBW:-0}
    if isinstance(intra, (int, float)) and actual > 0:
        ratio = actual / intra
        if ratio >= 0.9:
            print(f"  Status: ✓ Performance OK ({ratio:.1%} of expected)")
        elif ratio >= 0.7:
            print(f"  Status: ⚠ Performance Warning ({ratio:.1%} of expected)")
        else:
            print(f"  Status: ✗ Performance Issue ({ratio:.1%} of expected)")
else:
    print(f"  No baseline data for {gpu_model}")
PYTHON_SCRIPT
else
    echo "Baseline comparison not available"
fi

echo ""
echo "Results saved to:"
echo "  Summary: $OUTPUT_DIR/nccl_summary_${TIMESTAMP}.json"
echo "  Logs: $OUTPUT_DIR/*_${TIMESTAMP}.txt"
echo ""

# Recommendations
echo "Recommendations:"
if [ "${ALLREDUCE_BUSBW:-0}" != "0" ]; then
    if (( $(echo "${ALLREDUCE_BUSBW:-0} < 50" | bc -l) )); then
        echo "  - Check GPU topology: nvidia-smi topo -m"
        echo "  - Ensure NVLink is enabled (if available)"
        echo "  - Check for PCIe bottlenecks"
        echo "  - Review NUMA configuration"
    else
        echo "  - Performance looks good!"
    fi
fi

echo ""
echo "For multi-node testing:"
echo "  mpirun -np \$TOTAL_GPUS -N \$GPUS_PER_NODE \\"
echo "    --hostfile hostfile \\"
echo "    $NCCL_TESTS_DIR/all_reduce_perf -b 8 -e 8G -f 2 -g 1"
echo ""

echo "NCCL benchmarking complete!"

#!/bin/bash
# Megatron-LM Training Benchmark Script
# Tests GPT training throughput and compares against performance baselines

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
OUTPUT_DIR="${1:-/tmp/megatron_benchmarks}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MEGATRON_DIR="${MEGATRON_DIR:-/opt/Megatron-LM}"

# Model configuration options
MODEL_SIZE="${MODEL_SIZE:-GPT-1.2B}"  # GPT-1.2B, GPT-8.3B, or custom
BATCH_SIZE="${BATCH_SIZE:-8}"
SEQ_LENGTH="${SEQ_LENGTH:-2048}"
NUM_STEPS="${NUM_STEPS:-100}"  # Number of training steps for benchmark

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Megatron-LM Training Benchmark"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Timestamp: $TIMESTAMP"
echo ""

# Check if Megatron-LM is available
if [ ! -d "$MEGATRON_DIR" ]; then
    echo -e "${YELLOW}WARNING: Megatron-LM not found at $MEGATRON_DIR${NC}"
    echo ""
    echo "To install Megatron-LM:"
    echo "  git clone https://github.com/NVIDIA/Megatron-LM.git"
    echo "  cd Megatron-LM"
    echo "  pip install -r requirements.txt"
    echo ""
    echo "Then set MEGATRON_DIR=/path/to/Megatron-LM"
    echo ""
    echo "This script will create a sample training configuration."
fi

# Detect system configuration
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -d ' ')
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)

echo "System Configuration:"
echo "  GPU Model: $GPU_MODEL"
echo "  GPU Count: $GPU_COUNT"
echo "  GPU Memory: ${GPU_MEMORY} MB"
echo "  Model: $MODEL_SIZE"
echo "  Batch Size: $BATCH_SIZE"
echo "  Sequence Length: $SEQ_LENGTH"
echo "  Training Steps: $NUM_STEPS"
echo ""

# Model parameters based on size
case $MODEL_SIZE in
    "GPT-1.2B")
        NUM_LAYERS=24
        HIDDEN_SIZE=2048
        NUM_ATTENTION_HEADS=32
        ;;
    "GPT-8.3B")
        NUM_LAYERS=72
        HIDDEN_SIZE=3072
        NUM_ATTENTION_HEADS=24
        ;;
    "GPT-175B")
        NUM_LAYERS=96
        HIDDEN_SIZE=12288
        NUM_ATTENTION_HEADS=96
        ;;
    *)
        echo "Unknown model size: $MODEL_SIZE"
        echo "Supported: GPT-1.2B, GPT-8.3B, GPT-175B"
        exit 1
        ;;
esac

echo "Model Parameters:"
echo "  Layers: $NUM_LAYERS"
echo "  Hidden Size: $HIDDEN_SIZE"
echo "  Attention Heads: $NUM_ATTENTION_HEADS"
echo ""

#==========================================
# Prepare synthetic data for benchmarking
#==========================================
echo -e "${BLUE}Preparing benchmark environment...${NC}"

DATA_DIR="$OUTPUT_DIR/data"
mkdir -p "$DATA_DIR"

# Create a minimal training configuration script
cat > "$OUTPUT_DIR/benchmark_config.sh" << 'EOF'
#!/bin/bash
# Megatron-LM Benchmark Configuration

GPUS_PER_NODE=__GPU_COUNT__
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6000

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# Tensor and pipeline parallelism
if [ $GPUS_PER_NODE -ge 8 ]; then
    TP=4
    PP=2
elif [ $GPUS_PER_NODE -ge 4 ]; then
    TP=2
    PP=2
elif [ $GPUS_PER_NODE -ge 2 ]; then
    TP=2
    PP=1
else
    TP=1
    PP=1
fi

echo "Parallelism Configuration:"
echo "  Tensor Parallel: $TP"
echo "  Pipeline Parallel: $PP"
EOF

sed -i "s/__GPU_COUNT__/$GPU_COUNT/g" "$OUTPUT_DIR/benchmark_config.sh"
chmod +x "$OUTPUT_DIR/benchmark_config.sh"

#==========================================
# Run Training Benchmark
#==========================================
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Running Training Benchmark${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

if [ -d "$MEGATRON_DIR" ]; then
    # Create a minimal benchmark script
    cat > "$OUTPUT_DIR/run_benchmark.sh" << EOF
#!/bin/bash
source $OUTPUT_DIR/benchmark_config.sh

# Set GPU count based parallelism
if [ \$GPUS_PER_NODE -ge 8 ]; then
    TP=4; PP=2
elif [ \$GPUS_PER_NODE -ge 4 ]; then
    TP=2; PP=2
elif [ \$GPUS_PER_NODE -ge 2 ]; then
    TP=2; PP=1
else
    TP=1; PP=1
fi

cd $MEGATRON_DIR

python -m torch.distributed.launch \
    \$DISTRIBUTED_ARGS \
    pretrain_gpt.py \
    --num-layers $NUM_LAYERS \
    --hidden-size $HIDDEN_SIZE \
    --num-attention-heads $NUM_ATTENTION_HEADS \
    --micro-batch-size $BATCH_SIZE \
    --global-batch-size \$(($BATCH_SIZE * $GPU_COUNT)) \
    --seq-length $SEQ_LENGTH \
    --max-position-embeddings $SEQ_LENGTH \
    --train-iters $NUM_STEPS \
    --lr-decay-iters $NUM_STEPS \
    --data-path $DATA_DIR \
    --vocab-file $DATA_DIR/vocab.json \
    --merge-file $DATA_DIR/merges.txt \
    --lr 0.00015 \
    --min-lr 1.0e-5 \
    --lr-decay-style cosine \
    --lr-warmup-fraction .01 \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --fp16 \
    --tensor-model-parallel-size \$TP \
    --pipeline-model-parallel-size \$PP \
    --DDP-impl local \
    --log-interval 10 \
    --save-interval 10000 \
    --eval-interval 10000 \
    --eval-iters 10 \
    | tee $OUTPUT_DIR/training_log_${TIMESTAMP}.txt
EOF
    chmod +x "$OUTPUT_DIR/run_benchmark.sh"

    echo "Starting training benchmark..."
    echo "This may take several minutes..."
    echo ""

    # Run the benchmark
    "$OUTPUT_DIR/run_benchmark.sh" 2>&1 | tee "$OUTPUT_DIR/benchmark_output_${TIMESTAMP}.txt" &
    BENCHMARK_PID=$!

    # Monitor progress
    sleep 5
    if ps -p $BENCHMARK_PID > /dev/null; then
        echo "Benchmark running (PID: $BENCHMARK_PID)"
        echo "Monitor progress with: tail -f $OUTPUT_DIR/training_log_${TIMESTAMP}.txt"
        wait $BENCHMARK_PID
    else
        echo -e "${RED}Benchmark failed to start${NC}"
        echo "Check logs in $OUTPUT_DIR/"
    fi
else
    echo -e "${YELLOW}Megatron-LM not installed, skipping training benchmark${NC}"
    echo ""
    echo "Alternative: Run a simple PyTorch distributed training test"
    echo ""

    # Create a minimal PyTorch training benchmark
    cat > "$OUTPUT_DIR/simple_training_benchmark.py" << 'PYEOF'
#!/usr/bin/env python3
"""Simple distributed training benchmark"""
import torch
import torch.distributed as dist
import torch.nn as nn
import time
from datetime import datetime

def setup_distributed():
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(dist.get_rank())
    return dist.get_rank(), dist.get_world_size()

class SimpleModel(nn.Module):
    def __init__(self, hidden_size=2048):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x):
        return self.layers(x)

def benchmark():
    rank, world_size = setup_distributed()
    device = torch.cuda.current_device()

    print(f"[Rank {rank}] Running on GPU {device}")

    # Create model
    model = SimpleModel().cuda()
    model = nn.parallel.DistributedDataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Warmup
    for _ in range(10):
        x = torch.randn(32, 2048).cuda()
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Benchmark
    torch.cuda.synchronize()
    start_time = time.time()

    num_steps = 100
    for step in range(num_steps):
        x = torch.randn(32, 2048).cuda()
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if rank == 0 and step % 10 == 0:
            print(f"Step {step}/{num_steps}")

    torch.cuda.synchronize()
    end_time = time.time()

    if rank == 0:
        elapsed = end_time - start_time
        throughput = num_steps / elapsed
        print(f"\nBenchmark Results:")
        print(f"  Time: {elapsed:.2f} seconds")
        print(f"  Throughput: {throughput:.2f} steps/sec")
        print(f"  GPUs: {world_size}")

if __name__ == "__main__":
    benchmark()
PYEOF

    chmod +x "$OUTPUT_DIR/simple_training_benchmark.py"

    echo "Running simple distributed training benchmark..."
    python -m torch.distributed.launch \
        --nproc_per_node=$GPU_COUNT \
        "$OUTPUT_DIR/simple_training_benchmark.py" \
        2>&1 | tee "$OUTPUT_DIR/simple_benchmark_${TIMESTAMP}.txt"
fi

#==========================================
# Parse Results
#==========================================
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Benchmark Results${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Try to parse Megatron training log
if [ -f "$OUTPUT_DIR/training_log_${TIMESTAMP}.txt" ]; then
    # Extract TFLOPS and throughput
    TFLOPS=$(grep "TFLOPs" "$OUTPUT_DIR/training_log_${TIMESTAMP}.txt" | tail -1 | grep -oP 'TFLOPs: \K[0-9.]+' || echo "0")
    SAMPLES_PER_SEC=$(grep "throughput" "$OUTPUT_DIR/training_log_${TIMESTAMP}.txt" | tail -1 | grep -oP '[0-9.]+ samples/s' || echo "0")

    echo "Training Performance:"
    echo "  TFLOPS: ${TFLOPS}"
    echo "  Throughput: ${SAMPLES_PER_SEC}"

    # Calculate MFU (Model FLOP Utilization)
    # MFU = achieved_tflops / peak_tflops
    # This would need GPU-specific peak TFLOPS
fi

# Create summary JSON
cat > "$OUTPUT_DIR/benchmark_summary_${TIMESTAMP}.json" << EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "gpu_model": "$GPU_MODEL",
  "gpu_count": $GPU_COUNT,
  "model_config": {
    "model_size": "$MODEL_SIZE",
    "num_layers": $NUM_LAYERS,
    "hidden_size": $HIDDEN_SIZE,
    "num_attention_heads": $NUM_ATTENTION_HEADS,
    "batch_size": $BATCH_SIZE,
    "seq_length": $SEQ_LENGTH
  },
  "results": {
    "tflops": ${TFLOPS:-0},
    "throughput": "${SAMPLES_PER_SEC:-N/A}"
  }
}
EOF

#==========================================
# Compare Against Baselines
#==========================================
echo ""
echo "Baseline Comparison:"

BASELINE_SCRIPT="$(dirname $0)/../utils/performance_baselines.py"
if [ -f "$BASELINE_SCRIPT" ]; then
    python3 - <<PYTHON_SCRIPT
import sys
sys.path.insert(0, '$(dirname $BASELINE_SCRIPT)')
from performance_baselines import get_megatron_baseline, get_gpu_baseline

model_size = "$MODEL_SIZE"
gpu_model = "$GPU_MODEL"

megatron_baseline = get_megatron_baseline(model_size)
gpu_baseline = get_gpu_baseline(gpu_model)

print(f"Expected performance for {model_size}:")
if megatron_baseline:
    for config, metrics in megatron_baseline.items():
        if isinstance(metrics, dict) and 'tflops' in metrics:
            print(f"  {config}: {metrics['tflops']} TFLOPS")
else:
    print("  No baseline data available")

if gpu_baseline:
    print(f"\nGPU Peak Performance ({gpu_model}):")
    print(f"  FP16: {gpu_baseline.get('fp16_tflops', 'N/A')} TFLOPS")
    print(f"  TF32: {gpu_baseline.get('tf32_tflops', 'N/A')} TFLOPS")
    expected_mfu = gpu_baseline.get('expected_mfu', {}).get('megatron_gpt', 0)
    if expected_mfu:
        print(f"  Expected MFU: {expected_mfu*100:.1f}%")
PYTHON_SCRIPT
fi

echo ""
echo "Results saved to:"
echo "  Summary: $OUTPUT_DIR/benchmark_summary_${TIMESTAMP}.json"
echo "  Logs: $OUTPUT_DIR/*_${TIMESTAMP}.txt"
echo ""

echo "Megatron benchmark complete!"

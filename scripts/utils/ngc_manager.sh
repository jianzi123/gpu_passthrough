#!/bin/bash
# NGC Container Image Manager
# Convenient wrapper for pulling and running NGC images

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Help function
show_help() {
    cat << EOF
NGC Container Image Manager

Usage: $(basename "$0") <command> [options]

Commands:
  list                  List available NGC images in catalog
  pull <image> [ver]    Pull NGC image (image: pytorch, nemo, triton, etc.)
  run <image> [ver]     Run NGC image interactively
  test <image> [ver]    Test NGC image GPU functionality
  info <image> [ver]    Show image information
  catalog              Show full NGC image catalog
  cuda <version>       Show images compatible with CUDA version

Examples:
  $(basename "$0") list
  $(basename "$0") pull pytorch
  $(basename "$0") pull pytorch 24.01
  $(basename "$0") run nemo
  $(basename "$0") test pytorch
  $(basename "$0") cuda 12.3
  $(basename "$0") info triton

Environment Variables:
  NGC_API_KEY          NGC API key for private repositories (optional)
  CUDA_VISIBLE_DEVICES GPU devices to use (default: all)

Available Images:
  - pytorch    : PyTorch with CUDA, cuDNN, NCCL
  - tensorflow : TensorFlow with CUDA optimizations
  - nemo       : NeMo framework with Megatron-LM
  - triton     : Triton Inference Server
  - tensorrt   : TensorRT for inference
  - cuda       : CUDA development environment
  - rapids     : RAPIDS for data science
  - deepstream : DeepStream for video analytics

EOF
}

# Check if Python script exists
if [ ! -f "$SCRIPT_DIR/ngc_images.py" ]; then
    echo -e "${RED}ERROR: ngc_images.py not found${NC}"
    echo "Expected location: $SCRIPT_DIR/ngc_images.py"
    exit 1
fi

# List command
cmd_list() {
    echo -e "${BLUE}Available NGC Images:${NC}\n"
    python3 "$SCRIPT_DIR/ngc_images.py" --catalog | grep "^[A-Z]" | head -20
}

# Pull command
cmd_pull() {
    local image_name="$1"
    local version="${2:-}"

    echo -e "${BLUE}Pulling NGC image: $image_name${NC}"

    # Get pull command from Python script
    local pull_cmd
    pull_cmd=$(python3 "$SCRIPT_DIR/ngc_images.py" --pull "$image_name" "$version" 2>&1)

    if [ $? -ne 0 ]; then
        echo -e "${RED}ERROR: Failed to get pull command${NC}"
        echo "$pull_cmd"
        exit 1
    fi

    echo "Command: $pull_cmd"
    echo ""

    # Execute pull
    eval "$pull_cmd"

    if [ $? -eq 0 ]; then
        echo -e "\n${GREEN}Successfully pulled image${NC}"
    else
        echo -e "\n${RED}Failed to pull image${NC}"
        exit 1
    fi
}

# Run command
cmd_run() {
    local image_name="$1"
    local version="${2:-}"

    echo -e "${BLUE}Running NGC image: $image_name${NC}"

    # Get run command from Python script
    local run_cmd
    run_cmd=$(python3 "$SCRIPT_DIR/ngc_images.py" --run "$image_name" "$version" 2>&1)

    if [ $? -ne 0 ]; then
        echo -e "${RED}ERROR: Failed to get run command${NC}"
        echo "$run_cmd"
        exit 1
    fi

    echo "Command: $run_cmd"
    echo ""

    # Execute run with some common volume mounts
    local extra_args=""
    if [ -d "$HOME/workspace" ]; then
        extra_args="-v $HOME/workspace:/workspace"
    fi

    eval "$run_cmd $extra_args bash"
}

# Test command
cmd_test() {
    local image_name="$1"
    local version="${2:-}"

    echo -e "${BLUE}Testing NGC image: $image_name${NC}\n"

    # Get image URL
    local image_url
    image_url=$(python3 "$SCRIPT_DIR/ngc_images.py" "$image_name" "$version" 2>&1 | grep "URL:" | awk '{print $2}')

    if [ -z "$image_url" ]; then
        echo -e "${RED}ERROR: Failed to get image URL${NC}"
        exit 1
    fi

    echo "Image: $image_url"
    echo ""

    # Test based on image type
    case $image_name in
        pytorch|tensorflow|nemo)
            echo "Testing PyTorch/CUDA availability..."
            docker run --rm --gpus all "$image_url" \
                bash -c 'python3 -c "import torch; print(\"CUDA Available:\", torch.cuda.is_available()); print(\"GPU Count:\", torch.cuda.device_count()); print(\"GPU Name:\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\")"'
            ;;
        triton)
            echo "Testing Triton server..."
            docker run --rm "$image_url" tritonserver --help | head -5
            ;;
        tensorrt)
            echo "Testing TensorRT..."
            docker run --rm "$image_url" bash -c "python3 -c 'import tensorrt; print(\"TensorRT version:\", tensorrt.__version__)'"
            ;;
        *)
            echo "Running basic container test..."
            docker run --rm "$image_url" bash -c "echo 'Container started successfully'"
            ;;
    esac

    if [ $? -eq 0 ]; then
        echo -e "\n${GREEN}Test PASSED${NC}"
    else
        echo -e "\n${RED}Test FAILED${NC}"
        exit 1
    fi
}

# Info command
cmd_info() {
    local image_name="$1"
    local version="${2:-}"

    python3 "$SCRIPT_DIR/ngc_images.py" "$image_name" "$version"
}

# Catalog command
cmd_catalog() {
    python3 "$SCRIPT_DIR/ngc_images.py" --catalog
}

# CUDA compatibility command
cmd_cuda() {
    local cuda_version="$1"

    echo -e "${BLUE}NGC Images compatible with CUDA $cuda_version:${NC}\n"
    python3 "$SCRIPT_DIR/ngc_images.py" --cuda "$cuda_version"
}

# Main command dispatcher
if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

command="$1"
shift

case "$command" in
    list)
        cmd_list
        ;;
    pull)
        if [ $# -lt 1 ]; then
            echo "Usage: $(basename "$0") pull <image> [version]"
            exit 1
        fi
        cmd_pull "$@"
        ;;
    run)
        if [ $# -lt 1 ]; then
            echo "Usage: $(basename "$0") run <image> [version]"
            exit 1
        fi
        cmd_run "$@"
        ;;
    test)
        if [ $# -lt 1 ]; then
            echo "Usage: $(basename "$0") test <image> [version]"
            exit 1
        fi
        cmd_test "$@"
        ;;
    info)
        if [ $# -lt 1 ]; then
            echo "Usage: $(basename "$0") info <image> [version]"
            exit 1
        fi
        cmd_info "$@"
        ;;
    catalog)
        cmd_catalog
        ;;
    cuda)
        if [ $# -lt 1 ]; then
            echo "Usage: $(basename "$0") cuda <version>"
            exit 1
        fi
        cmd_cuda "$@"
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "Unknown command: $command"
        echo ""
        show_help
        exit 1
        ;;
esac

#!/bin/bash
# GPU Driver Installation Script
# Supports multiple installation methods: native, driver-container, precompiled
# Based on NVIDIA GPU Operator and GPU Driver Container architecture

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-/var/log/gpu_driver_install}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/install_$TIMESTAMP.log"

# Installation method: native, driver-container, precompiled
INSTALL_METHOD="${INSTALL_METHOD:-native}"

# Driver configuration
DRIVER_VERSION="${DRIVER_VERSION:-535.154.05}"
CUDA_VERSION="${CUDA_VERSION:-12.2}"
DRIVER_BRANCH="${DRIVER_BRANCH:-535}"

# Driver container configuration
DRIVER_CONTAINER_IMAGE="${DRIVER_CONTAINER_IMAGE:-nvcr.io/nvidia/driver}"
DRIVER_CONTAINER_TAG="${DRIVER_CONTAINER_TAG:-535.154.05-ubuntu22.04}"

# Precompiled driver configuration
USE_PRECOMPILED="${USE_PRECOMPILED:-false}"
KERNEL_VERSION=$(uname -r)
OS_ID=$(grep ^ID= /etc/os-release | cut -d= -f2 | tr -d '"')
OS_VERSION=$(grep ^VERSION_ID= /etc/os-release | cut -d= -f2 | tr -d '"')

# Auto-detect GPU
AUTO_DETECT_GPU="${AUTO_DETECT_GPU:-true}"

# Create log directory
mkdir -p "$LOG_DIR"

# Logging function
log() {
    echo -e "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log_info() {
    log "${BLUE}[INFO]${NC} $*"
}

log_success() {
    log "${GREEN}[SUCCESS]${NC} $*"
}

log_warn() {
    log "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    log "${RED}[ERROR]${NC} $*"
}

# Help function
show_help() {
    cat << EOF
GPU Driver Installation Script

Usage: $(basename "$0") [OPTIONS]

Installation Methods:
  native            Install driver directly on host (default)
  driver-container  Install driver using NVIDIA driver container
  precompiled       Use precompiled driver (faster, requires matching kernel)

Options:
  --method METHOD           Installation method (native|driver-container|precompiled)
  --driver-version VERSION  Driver version (default: 535.154.05)
  --cuda-version VERSION    CUDA version (default: 12.2)
  --auto-detect             Auto-detect GPU and select driver version (default: true)
  --precompiled             Use precompiled driver if available
  --container-image IMAGE   Driver container image (default: nvcr.io/nvidia/driver)
  --help                    Show this help message

Examples:
  # Native installation with auto-detection
  $(basename "$0") --method native --auto-detect

  # Driver container installation
  $(basename "$0") --method driver-container --driver-version 535.154.05

  # Precompiled driver installation
  $(basename "$0") --method precompiled --driver-version 535.154.05

Environment Variables:
  INSTALL_METHOD            Installation method
  DRIVER_VERSION            Driver version
  CUDA_VERSION              CUDA version
  AUTO_DETECT_GPU           Auto-detect GPU (true|false)
  USE_PRECOMPILED           Use precompiled drivers (true|false)
  LOG_DIR                   Log directory (default: /var/log/gpu_driver_install)

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --method)
            INSTALL_METHOD="$2"
            shift 2
            ;;
        --driver-version)
            DRIVER_VERSION="$2"
            shift 2
            ;;
        --cuda-version)
            CUDA_VERSION="$2"
            shift 2
            ;;
        --auto-detect)
            AUTO_DETECT_GPU=true
            shift
            ;;
        --precompiled)
            USE_PRECOMPILED=true
            shift
            ;;
        --container-image)
            DRIVER_CONTAINER_IMAGE="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Display configuration
log_info "=========================================="
log_info "GPU Driver Installation"
log_info "=========================================="
log_info "Installation Method: $INSTALL_METHOD"
log_info "Driver Version: $DRIVER_VERSION"
log_info "CUDA Version: $CUDA_VERSION"
log_info "Kernel Version: $KERNEL_VERSION"
log_info "OS: $OS_ID $OS_VERSION"
log_info "Auto-detect GPU: $AUTO_DETECT_GPU"
log_info "Use Precompiled: $USE_PRECOMPILED"
log_info "Log File: $LOG_FILE"
log_info "=========================================="
log_info ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    log_error "This script must be run as root"
    exit 1
fi

# Auto-detect GPU and select driver version
detect_gpu_and_driver() {
    log_info "Detecting GPU model..."

    if command -v lspci &> /dev/null; then
        GPU_MODEL=$(lspci | grep -i nvidia | grep -i "3D controller\|VGA compatible controller" | head -1 | sed 's/.*: //' | sed 's/ (rev.*//')
        log_info "Detected GPU: $GPU_MODEL"

        # Use cuda_compatibility.py if available
        if [ -f "$SCRIPT_DIR/../utils/cuda_compatibility.py" ]; then
            log_info "Using compatibility database to select driver version..."

            RECOMMENDED_CUDA=$(python3 "$SCRIPT_DIR/../utils/cuda_compatibility.py" "$GPU_MODEL" 2>/dev/null | grep "Recommended CUDA:" | awk '{print $3}' || echo "")
            RECOMMENDED_DRIVER=$(python3 "$SCRIPT_DIR/../utils/cuda_compatibility.py" "$GPU_MODEL" 2>/dev/null | grep "Recommended Driver:" | awk '{print $3}' || echo "")

            if [ -n "$RECOMMENDED_CUDA" ] && [ -n "$RECOMMENDED_DRIVER" ]; then
                CUDA_VERSION="$RECOMMENDED_CUDA"
                DRIVER_VERSION="$RECOMMENDED_DRIVER"
                DRIVER_BRANCH=$(echo "$DRIVER_VERSION" | cut -d. -f1)

                log_success "Selected CUDA: $CUDA_VERSION"
                log_success "Selected Driver: $DRIVER_VERSION"
            else
                log_warn "Could not auto-detect recommended versions, using defaults"
            fi
        fi
    else
        log_warn "lspci not found, cannot auto-detect GPU"
    fi
}

# Check for existing driver
check_existing_driver() {
    log_info "Checking for existing NVIDIA driver..."

    if command -v nvidia-smi &> /dev/null; then
        EXISTING_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 || echo "")

        if [ -n "$EXISTING_VERSION" ]; then
            log_warn "Existing NVIDIA driver found: $EXISTING_VERSION"

            if [ "$EXISTING_VERSION" == "$DRIVER_VERSION" ]; then
                log_success "Driver version matches target version: $DRIVER_VERSION"
                return 0
            else
                log_warn "Driver version differs from target: $DRIVER_VERSION"
                read -p "Do you want to continue and reinstall? (y/n) " -n 1 -r
                echo
                if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                    log_info "Installation cancelled by user"
                    exit 0
                fi
            fi
        fi
    else
        log_info "No existing NVIDIA driver found"
    fi

    return 1
}

# Disable nouveau driver
disable_nouveau() {
    log_info "Disabling nouveau driver..."

    cat > /etc/modprobe.d/blacklist-nouveau.conf << EOF
blacklist nouveau
options nouveau modeset=0
EOF

    if command -v update-initramfs &> /dev/null; then
        update-initramfs -u
    elif command -v dracut &> /dev/null; then
        dracut --force
    fi

    log_success "Nouveau driver disabled"
}

# Install prerequisites
install_prerequisites() {
    log_info "Installing prerequisites..."

    case $OS_ID in
        ubuntu|debian)
            apt-get update
            apt-get install -y \
                build-essential \
                dkms \
                linux-headers-$(uname -r) \
                pkg-config \
                libglvnd-dev \
                wget \
                curl \
                gnupg2 \
                ca-certificates
            ;;
        centos|rhel|rocky|almalinux)
            yum install -y \
                gcc \
                kernel-devel-$(uname -r) \
                kernel-headers-$(uname -r) \
                dkms \
                libglvnd-devel \
                wget \
                curl \
                gnupg2
            ;;
        *)
            log_error "Unsupported OS: $OS_ID"
            exit 1
            ;;
    esac

    log_success "Prerequisites installed"
}

# Native driver installation
install_native_driver() {
    log_info "Installing NVIDIA driver natively..."

    case $OS_ID in
        ubuntu|debian)
            # Add NVIDIA CUDA repository
            log_info "Adding NVIDIA CUDA repository..."

            CUDA_REPO_URL="https://developer.download.nvidia.com/compute/cuda/repos/${OS_ID}$(echo $OS_VERSION | tr -d '.')/x86_64"
            CUDA_REPO_KEY="${CUDA_REPO_URL}/3bf863cc.pub"

            wget -qO - "$CUDA_REPO_KEY" | apt-key add -

            echo "deb $CUDA_REPO_URL /" > /etc/apt/sources.list.d/cuda.list

            apt-get update

            # Install driver
            log_info "Installing driver version: $DRIVER_VERSION..."

            if [ "$DRIVER_BRANCH" ]; then
                apt-get install -y "nvidia-driver-$DRIVER_BRANCH"
            else
                apt-get install -y nvidia-driver
            fi
            ;;

        centos|rhel|rocky|almalinux)
            # Add NVIDIA CUDA repository
            log_info "Adding NVIDIA CUDA repository..."

            CUDA_REPO_RPM="https://developer.download.nvidia.com/compute/cuda/repos/rhel${OS_VERSION%%.*}/x86_64/cuda-repo-rhel${OS_VERSION%%.*}-${CUDA_VERSION//./-}.x86_64.rpm"

            yum install -y "$CUDA_REPO_RPM"
            yum clean all

            # Install driver
            log_info "Installing driver version: $DRIVER_VERSION..."
            yum install -y nvidia-driver-cuda
            ;;

        *)
            log_error "Unsupported OS for native installation: $OS_ID"
            exit 1
            ;;
    esac

    log_success "NVIDIA driver installed"
}

# Driver container installation
install_driver_container() {
    log_info "Installing NVIDIA driver using driver container..."

    # Ensure Docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    # Pull driver container
    DRIVER_IMAGE="${DRIVER_CONTAINER_IMAGE}:${DRIVER_CONTAINER_TAG}"
    log_info "Pulling driver container: $DRIVER_IMAGE"

    docker pull "$DRIVER_IMAGE"

    # Create systemd service for driver container
    log_info "Creating systemd service for driver container..."

    cat > /etc/systemd/system/nvidia-driver.service << EOF
[Unit]
Description=NVIDIA Driver Container
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStartPre=-/usr/bin/docker rm -f nvidia-driver
ExecStart=/usr/bin/docker run \\
    --name nvidia-driver \\
    --privileged \\
    --pid=host \\
    -v /run/nvidia:/run/nvidia:shared \\
    -v /var/log:/var/log \\
    --restart=unless-stopped \\
    -d \\
    ${DRIVER_IMAGE}

ExecStop=/usr/bin/docker stop nvidia-driver
ExecStopPost=/usr/bin/docker rm nvidia-driver

[Install]
WantedBy=multi-user.target
EOF

    # Enable and start service
    systemctl daemon-reload
    systemctl enable nvidia-driver.service
    systemctl start nvidia-driver.service

    # Wait for driver to load
    log_info "Waiting for driver to load..."
    for i in {1..60}; do
        if nvidia-smi &> /dev/null; then
            log_success "Driver loaded successfully"
            break
        fi
        sleep 2
    done

    log_success "Driver container installed and started"
}

# Precompiled driver installation
install_precompiled_driver() {
    log_info "Installing precompiled NVIDIA driver..."

    # Construct precompiled driver URL
    PRECOMPILED_URL="https://us.download.nvidia.com/tesla/${DRIVER_VERSION}/NVIDIA-Linux-x86_64-${DRIVER_VERSION}.run"

    log_info "Downloading precompiled driver from: $PRECOMPILED_URL"

    # Download driver
    wget -O "/tmp/NVIDIA-Linux-x86_64-${DRIVER_VERSION}.run" "$PRECOMPILED_URL"

    # Make executable
    chmod +x "/tmp/NVIDIA-Linux-x86_64-${DRIVER_VERSION}.run"

    # Install driver
    log_info "Installing driver (this may take several minutes)..."
    "/tmp/NVIDIA-Linux-x86_64-${DRIVER_VERSION}.run" \
        --silent \
        --no-questions \
        --ui=none \
        --disable-nouveau \
        --dkms

    # Clean up
    rm -f "/tmp/NVIDIA-Linux-x86_64-${DRIVER_VERSION}.run"

    log_success "Precompiled driver installed"
}

# Validate driver installation
validate_driver() {
    log_info "Validating driver installation..."

    # Wait for driver to be available
    sleep 5

    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi not found after installation"
        return 1
    fi

    # Run nvidia-smi
    log_info "Running nvidia-smi..."
    if nvidia-smi; then
        INSTALLED_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
        log_success "Driver validation successful"
        log_success "Installed driver version: $INSTALLED_VERSION"

        # Save validation report
        nvidia-smi > "$LOG_DIR/validation_$TIMESTAMP.txt"

        return 0
    else
        log_error "Driver validation failed"
        return 1
    fi
}

# Install NVIDIA Persistence Daemon
install_persistence_daemon() {
    log_info "Installing NVIDIA Persistence Daemon..."

    # Enable persistence mode
    nvidia-smi -pm 1

    # Create systemd service
    cat > /etc/systemd/system/nvidia-persistenced.service << 'EOF'
[Unit]
Description=NVIDIA Persistence Daemon
Wants=syslog.target

[Service]
Type=forking
ExecStart=/usr/bin/nvidia-persistenced --user root --persistence-mode
ExecStopPost=/bin/rm -rf /var/run/nvidia-persistenced
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable nvidia-persistenced.service
    systemctl start nvidia-persistenced.service

    log_success "NVIDIA Persistence Daemon installed"
}

# Generate installation report
generate_report() {
    local report_file="$LOG_DIR/installation_report_$TIMESTAMP.txt"

    cat > "$report_file" << EOF
NVIDIA Driver Installation Report
==================================
Timestamp: $(date)
Installation Method: $INSTALL_METHOD
Driver Version: $DRIVER_VERSION
CUDA Version: $CUDA_VERSION

System Information:
-------------------
OS: $OS_ID $OS_VERSION
Kernel: $KERNEL_VERSION
Architecture: $(uname -m)

GPU Information:
----------------
$(nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv)

Driver Validation:
------------------
$(nvidia-smi)

Installation Log:
-----------------
Log file: $LOG_FILE

EOF

    log_success "Installation report generated: $report_file"
    cat "$report_file"
}

# Main installation flow
main() {
    log_info "Starting GPU driver installation..."

    # Auto-detect GPU if enabled
    if [ "$AUTO_DETECT_GPU" == "true" ]; then
        detect_gpu_and_driver
    fi

    # Check for existing driver
    if check_existing_driver; then
        log_info "Driver already installed and matches target version"
        validate_driver
        exit 0
    fi

    # Disable nouveau
    disable_nouveau

    # Install prerequisites
    if [ "$INSTALL_METHOD" != "driver-container" ]; then
        install_prerequisites
    fi

    # Install driver based on method
    case $INSTALL_METHOD in
        native)
            install_native_driver
            ;;
        driver-container)
            install_driver_container
            ;;
        precompiled)
            install_precompiled_driver
            ;;
        *)
            log_error "Unknown installation method: $INSTALL_METHOD"
            show_help
            exit 1
            ;;
    esac

    # Validate installation
    if validate_driver; then
        # Install persistence daemon
        if [ "$INSTALL_METHOD" != "driver-container" ]; then
            install_persistence_daemon
        fi

        # Generate report
        generate_report

        log_success "=========================================="
        log_success "GPU Driver installation completed successfully!"
        log_success "Driver Version: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
        log_success "Please reboot the system if this is a new installation"
        log_success "=========================================="
    else
        log_error "Driver installation failed. Check log: $LOG_FILE"
        exit 1
    fi
}

# Run main function
main

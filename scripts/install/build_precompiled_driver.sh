#!/bin/bash
# Build Precompiled Driver Script
# Creates precompiled driver packages for faster deployment
# Based on NVIDIA GPU Operator precompiled driver architecture

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
DRIVER_VERSION="${DRIVER_VERSION:-535.154.05}"
KERNEL_VERSION="${KERNEL_VERSION:-$(uname -r)}"
BUILD_DIR="${BUILD_DIR:-/tmp/driver_build}"
OUTPUT_DIR="${OUTPUT_DIR:-/opt/precompiled_drivers}"
CONTAINER_BUILD="${CONTAINER_BUILD:-false}"

# Docker configuration for container build
DOCKER_IMAGE="${DOCKER_IMAGE:-nvidia/driver-build}"
DOCKER_TAG="${DOCKER_TAG:-ubuntu22.04}"

# Logging
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

# Help function
show_help() {
    cat << EOF
Build Precompiled Driver Script

Usage: $(basename "$0") [OPTIONS]

Options:
  --driver-version VERSION  Driver version to build (default: 535.154.05)
  --kernel-version VERSION  Kernel version to build for (default: current kernel)
  --build-dir DIR           Build directory (default: /tmp/driver_build)
  --output-dir DIR          Output directory (default: /opt/precompiled_drivers)
  --container-build         Build in Docker container (cleaner, recommended)
  --help                    Show this help message

Examples:
  # Build for current kernel
  $(basename "$0") --driver-version 535.154.05

  # Build in container
  $(basename "$0") --driver-version 535.154.05 --container-build

  # Build for specific kernel
  $(basename "$0") --driver-version 535.154.05 --kernel-version 5.15.0-91-generic

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --driver-version)
            DRIVER_VERSION="$2"
            shift 2
            ;;
        --kernel-version)
            KERNEL_VERSION="$2"
            shift 2
            ;;
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --container-build)
            CONTAINER_BUILD=true
            shift
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

log_info "=========================================="
log_info "Precompiled Driver Build"
log_info "=========================================="
log_info "Driver Version: $DRIVER_VERSION"
log_info "Kernel Version: $KERNEL_VERSION"
log_info "Build Directory: $BUILD_DIR"
log_info "Output Directory: $OUTPUT_DIR"
log_info "Container Build: $CONTAINER_BUILD"
log_info "=========================================="
log_info ""

# Create directories
mkdir -p "$BUILD_DIR"
mkdir -p "$OUTPUT_DIR"

# Download driver source
download_driver() {
    log_info "Downloading driver source..."

    local driver_url="https://us.download.nvidia.com/tesla/${DRIVER_VERSION}/NVIDIA-Linux-x86_64-${DRIVER_VERSION}.run"
    local driver_file="$BUILD_DIR/NVIDIA-Linux-x86_64-${DRIVER_VERSION}.run"

    if [ -f "$driver_file" ]; then
        log_info "Driver already downloaded"
    else
        wget -O "$driver_file" "$driver_url"
        chmod +x "$driver_file"
    fi

    log_success "Driver downloaded: $driver_file"
}

# Extract driver
extract_driver() {
    log_info "Extracting driver..."

    local driver_file="$BUILD_DIR/NVIDIA-Linux-x86_64-${DRIVER_VERSION}.run"
    local extract_dir="$BUILD_DIR/extracted"

    rm -rf "$extract_dir"
    "$driver_file" --extract-only --target "$extract_dir"

    log_success "Driver extracted to: $extract_dir"
}

# Build driver modules
build_driver_native() {
    log_info "Building driver modules for kernel: $KERNEL_VERSION..."

    local extract_dir="$BUILD_DIR/extracted"
    local kernel_dir="/lib/modules/$KERNEL_VERSION/build"

    if [ ! -d "$kernel_dir" ]; then
        log_error "Kernel headers not found for: $KERNEL_VERSION"
        log_error "Please install: linux-headers-$KERNEL_VERSION"
        exit 1
    fi

    cd "$extract_dir/kernel"

    # Build modules
    make -j$(nproc) \
        SYSSRC="$kernel_dir" \
        module

    log_success "Driver modules built successfully"
}

# Package precompiled driver
package_driver() {
    log_info "Packaging precompiled driver..."

    local extract_dir="$BUILD_DIR/extracted"
    local package_name="nvidia-driver-${DRIVER_VERSION}-kernel-${KERNEL_VERSION}"
    local package_dir="$BUILD_DIR/$package_name"
    local output_file="$OUTPUT_DIR/${package_name}.tar.gz"

    # Create package directory
    rm -rf "$package_dir"
    mkdir -p "$package_dir"/{modules,firmware}

    # Copy built modules
    find "$extract_dir/kernel" -name "*.ko" -exec cp {} "$package_dir/modules/" \;

    # Copy firmware if exists
    if [ -d "$extract_dir/firmware" ]; then
        cp -r "$extract_dir/firmware/"* "$package_dir/firmware/" || true
    fi

    # Create metadata
    cat > "$package_dir/metadata.json" << EOF
{
  "driver_version": "$DRIVER_VERSION",
  "kernel_version": "$KERNEL_VERSION",
  "build_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "architecture": "x86_64"
}
EOF

    # Create installation script
    cat > "$package_dir/install.sh" << 'INSTALL_EOF'
#!/bin/bash
# Precompiled driver installation script

set -e

KERNEL_VERSION=$(uname -r)
MODULE_DIR="/lib/modules/$KERNEL_VERSION/kernel/drivers/video"

echo "Installing NVIDIA precompiled driver..."

# Create module directory
mkdir -p "$MODULE_DIR"

# Copy modules
cp modules/*.ko "$MODULE_DIR/"

# Update module dependencies
depmod -a

# Load modules
modprobe nvidia
modprobe nvidia-uvm

echo "Driver installed successfully"
echo "Run 'nvidia-smi' to verify installation"
INSTALL_EOF

    chmod +x "$package_dir/install.sh"

    # Create archive
    tar -czf "$output_file" -C "$BUILD_DIR" "$package_name"

    log_success "Precompiled driver package created: $output_file"

    # Create checksum
    sha256sum "$output_file" > "$output_file.sha256"

    log_success "Checksum created: $output_file.sha256"
}

# Build in Docker container
build_in_container() {
    log_info "Building driver in Docker container..."

    # Create Dockerfile
    cat > "$BUILD_DIR/Dockerfile" << EOF
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    dkms \\
    linux-headers-$KERNEL_VERSION \\
    wget \\
    ca-certificates

WORKDIR /build

# Copy driver
COPY NVIDIA-Linux-x86_64-${DRIVER_VERSION}.run /build/

# Build driver
RUN chmod +x NVIDIA-Linux-x86_64-${DRIVER_VERSION}.run && \\
    ./NVIDIA-Linux-x86_64-${DRIVER_VERSION}.run \\
        --extract-only \\
        --target /build/extracted

WORKDIR /build/extracted/kernel

RUN make -j\$(nproc) \\
    SYSSRC=/lib/modules/$KERNEL_VERSION/build \\
    module

# Package modules
RUN mkdir -p /output/modules && \\
    find /build/extracted/kernel -name "*.ko" -exec cp {} /output/modules/ \;

VOLUME /output
EOF

    # Build Docker image
    log_info "Building Docker image..."
    docker build -t "${DOCKER_IMAGE}:${DOCKER_TAG}" "$BUILD_DIR"

    # Run container to build and extract
    log_info "Running build container..."
    docker run --rm \
        -v "$OUTPUT_DIR:/output" \
        "${DOCKER_IMAGE}:${DOCKER_TAG}" \
        bash -c "cp -r /output/modules /host_output/"

    log_success "Container build completed"
}

# Validate precompiled driver
validate_package() {
    log_info "Validating precompiled driver package..."

    local package_name="nvidia-driver-${DRIVER_VERSION}-kernel-${KERNEL_VERSION}"
    local output_file="$OUTPUT_DIR/${package_name}.tar.gz"

    if [ ! -f "$output_file" ]; then
        log_error "Package not found: $output_file"
        return 1
    fi

    # Verify checksum
    if [ -f "$output_file.sha256" ]; then
        cd "$OUTPUT_DIR"
        if sha256sum -c "${package_name}.tar.gz.sha256"; then
            log_success "Checksum verification passed"
        else
            log_error "Checksum verification failed"
            return 1
        fi
    fi

    # List package contents
    log_info "Package contents:"
    tar -tzf "$output_file" | head -20

    # Check for required files
    if tar -tzf "$output_file" | grep -q "modules/nvidia.ko"; then
        log_success "Required modules found in package"
    else
        log_error "Required modules missing from package"
        return 1
    fi

    log_success "Package validation completed"
}

# Generate build report
generate_report() {
    local report_file="$OUTPUT_DIR/build_report_$(date +%Y%m%d_%H%M%S).txt"

    cat > "$report_file" << EOF
Precompiled Driver Build Report
================================
Build Date: $(date)
Driver Version: $DRIVER_VERSION
Kernel Version: $KERNEL_VERSION
Build Method: $([ "$CONTAINER_BUILD" == "true" ] && echo "Container" || echo "Native")

Output Files:
-------------
$(ls -lh "$OUTPUT_DIR"/*.tar.gz)

Package Contents:
-----------------
$(tar -tzf "$OUTPUT_DIR/nvidia-driver-${DRIVER_VERSION}-kernel-${KERNEL_VERSION}.tar.gz")

Build System:
-------------
OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d= -f2 | tr -d '"')
Kernel: $(uname -r)
Architecture: $(uname -m)

EOF

    log_success "Build report generated: $report_file"
    cat "$report_file"
}

# Main function
main() {
    log_info "Starting precompiled driver build..."

    # Download driver
    download_driver

    if [ "$CONTAINER_BUILD" == "true" ]; then
        # Build in container
        build_in_container
    else
        # Build natively
        extract_driver
        build_driver_native
    fi

    # Package driver
    package_driver

    # Validate package
    validate_package

    # Generate report
    generate_report

    log_success "=========================================="
    log_success "Precompiled driver build completed!"
    log_success "Package: $OUTPUT_DIR/nvidia-driver-${DRIVER_VERSION}-kernel-${KERNEL_VERSION}.tar.gz"
    log_success "=========================================="
}

# Run main
main

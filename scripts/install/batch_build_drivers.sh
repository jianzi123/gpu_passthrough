#!/bin/bash
# Batch Build Precompiled Drivers
# Build drivers for multiple kernel versions and driver versions

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-/opt/precompiled-drivers}"
BUILD_LOG="${OUTPUT_DIR}/build.log"
USE_CONTAINER="${USE_CONTAINER:-true}"

# Driver versions to build
DRIVER_VERSIONS=(
    "535.154.05"
    "550.90.07"
)

# Kernel versions to build for
KERNEL_VERSIONS=(
    "5.15.0-91-generic"
    "5.15.0-92-generic"
    "5.15.0-94-generic"
    "6.2.0-39-generic"
    "6.5.0-14-generic"
)

# Logging
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*" | tee -a "$BUILD_LOG"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*" | tee -a "$BUILD_LOG"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*" | tee -a "$BUILD_LOG"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" | tee -a "$BUILD_LOG"
}

# Initialize
mkdir -p "$OUTPUT_DIR"
echo "=== Build started at $(date) ===" > "$BUILD_LOG"

# Build summary
TOTAL_BUILDS=0
SUCCESSFUL_BUILDS=0
FAILED_BUILDS=0
BUILD_RESULTS=()

# Build single driver
build_driver() {
    local driver_ver=$1
    local kernel_ver=$2

    log_info "=========================================="
    log_info "Building: Driver $driver_ver for Kernel $kernel_ver"
    log_info "=========================================="

    TOTAL_BUILDS=$((TOTAL_BUILDS + 1))

    # Check if already exists
    local output_file="nvidia-driver-${driver_ver}-kernel-${kernel_ver}.tar.gz"
    if [ -f "${OUTPUT_DIR}/${output_file}" ]; then
        log_warn "Package already exists, skipping: $output_file"
        return 0
    fi

    # Build command
    local build_cmd="./scripts/install/build_precompiled_driver.sh \
        --driver-version ${driver_ver} \
        --kernel-version ${kernel_ver} \
        --output-dir ${OUTPUT_DIR}"

    if [ "$USE_CONTAINER" = "true" ]; then
        build_cmd="$build_cmd --container-build"
    fi

    # Execute build
    if $build_cmd >> "$BUILD_LOG" 2>&1; then
        log_success "✓ Build successful: $output_file"
        SUCCESSFUL_BUILDS=$((SUCCESSFUL_BUILDS + 1))
        BUILD_RESULTS+=("SUCCESS|$driver_ver|$kernel_ver")
    else
        log_error "✗ Build failed: $output_file"
        FAILED_BUILDS=$((FAILED_BUILDS + 1))
        BUILD_RESULTS+=("FAILED|$driver_ver|$kernel_ver")
    fi

    echo ""
}

# Main build loop
log_info "Starting batch build..."
log_info "Driver versions: ${DRIVER_VERSIONS[*]}"
log_info "Kernel versions: ${KERNEL_VERSIONS[*]}"
log_info "Output directory: $OUTPUT_DIR"
log_info "Container build: $USE_CONTAINER"
echo ""

START_TIME=$(date +%s)

# Build all combinations
for driver in "${DRIVER_VERSIONS[@]}"; do
    for kernel in "${KERNEL_VERSIONS[@]}"; do
        build_driver "$driver" "$kernel"
    done
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Generate summary
log_info "=========================================="
log_info "Build Summary"
log_info "=========================================="
log_info "Total builds: $TOTAL_BUILDS"
log_success "Successful: $SUCCESSFUL_BUILDS"
log_error "Failed: $FAILED_BUILDS"
log_info "Duration: ${DURATION} seconds"
echo ""

# Detailed results
log_info "Build Results:"
printf "%-10s %-20s %-25s\n" "STATUS" "DRIVER" "KERNEL"
printf "%s\n" "-----------------------------------------------------------"

for result in "${BUILD_RESULTS[@]}"; do
    IFS='|' read -r status driver kernel <<< "$result"

    if [ "$status" = "SUCCESS" ]; then
        printf "${GREEN}%-10s${NC} %-20s %-25s\n" "$status" "$driver" "$kernel"
    else
        printf "${RED}%-10s${NC} %-20s %-25s\n" "$status" "$driver" "$kernel"
    fi
done

echo ""

# List all packages
log_info "Created packages:"
ls -lh "${OUTPUT_DIR}"/*.tar.gz 2>/dev/null || log_warn "No packages found"

echo ""

# Calculate total size
if ls "${OUTPUT_DIR}"/*.tar.gz &>/dev/null; then
    TOTAL_SIZE=$(du -sh "${OUTPUT_DIR}" | cut -f1)
    log_info "Total size: $TOTAL_SIZE"
fi

# Update index
log_info "Updating repository index..."
if [ -f "./scripts/utils/manage_precompiled_drivers.sh" ]; then
    ./scripts/utils/manage_precompiled_drivers.sh update-index
fi

# Create build report
REPORT_FILE="${OUTPUT_DIR}/build_report_$(date +%Y%m%d_%H%M%S).txt"

cat > "$REPORT_FILE" << EOF
Batch Build Report
==================
Build Date: $(date)
Duration: ${DURATION} seconds

Configuration:
--------------
Driver Versions: ${DRIVER_VERSIONS[*]}
Kernel Versions: ${KERNEL_VERSIONS[*]}
Container Build: $USE_CONTAINER
Output Directory: $OUTPUT_DIR

Results:
--------
Total Builds: $TOTAL_BUILDS
Successful: $SUCCESSFUL_BUILDS
Failed: $FAILED_BUILDS
Success Rate: $((SUCCESSFUL_BUILDS * 100 / TOTAL_BUILDS))%

Packages Created:
-----------------
$(ls -lh ${OUTPUT_DIR}/*.tar.gz 2>/dev/null || echo "None")

Total Size: $(du -sh ${OUTPUT_DIR} | cut -f1)

Detailed Results:
-----------------
EOF

for result in "${BUILD_RESULTS[@]}"; do
    IFS='|' read -r status driver kernel <<< "$result"
    echo "$status - Driver: $driver, Kernel: $kernel" >> "$REPORT_FILE"
done

log_success "Build report saved: $REPORT_FILE"

# Exit with error if any builds failed
if [ $FAILED_BUILDS -gt 0 ]; then
    log_error "Some builds failed. Check log: $BUILD_LOG"
    exit 1
else
    log_success "All builds completed successfully!"
    exit 0
fi

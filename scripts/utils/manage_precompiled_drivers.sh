#!/bin/bash
# Precompiled Driver Management Script
# Unified tool for managing precompiled NVIDIA drivers

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
REPO_DIR="${DRIVER_REPO_DIR:-/opt/precompiled-drivers}"
CACHE_DIR="/var/cache/nvidia-drivers"
STATE_FILE="/var/lib/nvidia/driver_state.json"

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
Precompiled Driver Management Tool

Usage: $(basename "$0") <command> [options]

Commands:
  list                      List all available precompiled drivers
  list-installed            Show currently installed driver
  install <version>         Install specific driver version
  install-latest            Install latest available driver
  uninstall                 Uninstall current driver
  rollback                  Rollback to previous version
  verify                    Verify current installation
  clean                     Clean old/unused packages
  update-index              Update driver repository index
  show <version>            Show driver package details
  download <version>        Download driver without installing
  search <kernel>           Search drivers for specific kernel

Options:
  --repo <path>             Repository directory (default: $REPO_DIR)
  --force                   Force installation
  --no-backup               Skip backup of current driver
  --quiet                   Minimal output

Examples:
  # List available drivers
  $(basename "$0") list

  # Install specific version
  $(basename "$0") install 535.154.05

  # Install latest driver
  $(basename "$0") install-latest

  # Rollback to previous version
  $(basename "$0") rollback

  # Search for drivers compatible with current kernel
  $(basename "$0") search \$(uname -r)

  # Clean old packages
  $(basename "$0") clean

EOF
}

# Initialize environment
init_environment() {
    mkdir -p "$REPO_DIR" "$CACHE_DIR" "$(dirname "$STATE_FILE")"
}

# List all available drivers
list_drivers() {
    log_info "Available precompiled drivers in $REPO_DIR:"
    echo ""

    if [ ! -d "$REPO_DIR" ]; then
        log_error "Repository directory not found: $REPO_DIR"
        return 1
    fi

    # Find all driver packages
    local packages=$(find "$REPO_DIR" -name "nvidia-driver-*.tar.gz" -type f | sort -V)

    if [ -z "$packages" ]; then
        log_warn "No precompiled drivers found"
        return 0
    fi

    # Display in table format
    printf "%-20s %-20s %-10s %-15s\n" "DRIVER VERSION" "KERNEL VERSION" "SIZE" "DATE"
    printf "%s\n" "--------------------------------------------------------------------------------"

    while IFS= read -r package; do
        local basename=$(basename "$package" .tar.gz)

        # Extract version information
        if [[ $basename =~ nvidia-driver-([0-9.]+)-kernel-([^ ]+) ]]; then
            local driver_ver="${BASH_REMATCH[1]}"
            local kernel_ver="${BASH_REMATCH[2]}"
            local size=$(du -h "$package" | cut -f1)
            local date=$(stat -c %y "$package" | cut -d' ' -f1)

            printf "%-20s %-20s %-10s %-15s\n" "$driver_ver" "$kernel_ver" "$size" "$date"
        fi
    done <<< "$packages"

    echo ""
    log_info "Total packages: $(echo "$packages" | wc -l)"
}

# Show currently installed driver
list_installed() {
    log_info "Currently installed NVIDIA driver:"
    echo ""

    if command -v nvidia-smi &>/dev/null; then
        local version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
        local kernel=$(uname -r)

        printf "%-20s: %s\n" "Driver Version" "$version"
        printf "%-20s: %s\n" "Kernel Version" "$kernel"
        printf "%-20s: %s\n" "CUDA Version" "$(nvidia-smi | grep "CUDA Version:" | awk '{print $9}')"

        # Check state file
        if [ -f "$STATE_FILE" ]; then
            echo ""
            log_info "Installation history:"
            cat "$STATE_FILE" | jq -r '.history[] | "\(.date) - \(.version)"' | tail -5
        fi
    else
        log_warn "No NVIDIA driver installed"
        return 1
    fi
}

# Find driver package
find_driver_package() {
    local version=$1
    local kernel=${2:-$(uname -r)}

    # Search for exact match
    local package=$(find "$REPO_DIR" -name "nvidia-driver-${version}-kernel-${kernel}*.tar.gz" -type f | head -1)

    if [ -n "$package" ]; then
        echo "$package"
        return 0
    fi

    # Try partial kernel match
    package=$(find "$REPO_DIR" -name "nvidia-driver-${version}-kernel-${kernel%%-*}*.tar.gz" -type f | head -1)

    if [ -n "$package" ]; then
        log_warn "Exact kernel match not found, using: $(basename $package)"
        echo "$package"
        return 0
    fi

    return 1
}

# Save driver state
save_state() {
    local version=$1
    local action=$2

    local state="{
        \"version\": \"$version\",
        \"kernel\": \"$(uname -r)\",
        \"date\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",
        \"action\": \"$action\",
        \"hostname\": \"$(hostname)\"
    }"

    # Create or update state file
    if [ -f "$STATE_FILE" ]; then
        jq ".history += [$state]" "$STATE_FILE" > "${STATE_FILE}.tmp"
        mv "${STATE_FILE}.tmp" "$STATE_FILE"
    else
        echo "{\"history\": [$state]}" > "$STATE_FILE"
    fi
}

# Backup current driver
backup_current_driver() {
    if ! command -v nvidia-smi &>/dev/null; then
        log_info "No driver to backup"
        return 0
    fi

    local version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    local backup_dir="/var/lib/nvidia/backup/${version}-$(date +%Y%m%d)"

    log_info "Backing up current driver (version $version)..."

    mkdir -p "$backup_dir"

    # Backup module files
    local kernel=$(uname -r)
    if [ -d "/lib/modules/$kernel/kernel/drivers/video" ]; then
        cp -a "/lib/modules/$kernel/kernel/drivers/video"/nvidia*.ko "$backup_dir/" 2>/dev/null || true
    fi

    log_success "Driver backed up to: $backup_dir"
}

# Install driver
install_driver() {
    local version=$1
    local force=${2:-false}
    local no_backup=${3:-false}

    log_info "Installing precompiled driver version: $version"

    # Find package
    local package=$(find_driver_package "$version")

    if [ -z "$package" ]; then
        log_error "Driver package not found for version: $version"
        log_info "Run '$(basename $0) list' to see available drivers"
        return 1
    fi

    log_info "Found package: $(basename $package)"

    # Check if already installed
    if command -v nvidia-smi &>/dev/null; then
        local current=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)

        if [ "$current" == "$version" ] && [ "$force" != "true" ]; then
            log_warn "Driver version $version is already installed"
            log_info "Use --force to reinstall"
            return 0
        fi

        # Backup current driver
        if [ "$no_backup" != "true" ]; then
            backup_current_driver
        fi

        # Unload current driver
        log_info "Unloading current driver..."
        rmmod nvidia_drm nvidia_modeset nvidia_uvm nvidia 2>/dev/null || true
    fi

    # Extract and install
    local temp_dir=$(mktemp -d)
    log_info "Extracting package..."

    tar -xzf "$package" -C "$temp_dir"
    cd "$temp_dir"/*

    # Run installation script
    log_info "Running installation..."
    if [ -x "./install.sh" ]; then
        ./install.sh
    else
        log_error "Installation script not found or not executable"
        rm -rf "$temp_dir"
        return 1
    fi

    # Verify installation
    if nvidia-smi &>/dev/null; then
        log_success "Driver installed successfully!"

        # Save state
        save_state "$version" "install"

        # Display info
        nvidia-smi --query-gpu=gpu_name,driver_version,memory.total --format=csv
    else
        log_error "Driver installation failed - nvidia-smi not working"
        rm -rf "$temp_dir"
        return 1
    fi

    # Cleanup
    rm -rf "$temp_dir"
}

# Install latest driver
install_latest() {
    log_info "Finding latest driver for kernel $(uname -r)..."

    local kernel=$(uname -r)
    local latest=$(find "$REPO_DIR" -name "nvidia-driver-*-kernel-${kernel}*.tar.gz" -type f | \
                   sort -V | tail -1)

    if [ -z "$latest" ]; then
        log_error "No driver found for kernel: $kernel"
        return 1
    fi

    # Extract version
    local basename=$(basename "$latest" .tar.gz)
    if [[ $basename =~ nvidia-driver-([0-9.]+)-kernel- ]]; then
        local version="${BASH_REMATCH[1]}"
        log_info "Latest version: $version"
        install_driver "$version"
    else
        log_error "Failed to parse version from: $basename"
        return 1
    fi
}

# Uninstall driver
uninstall_driver() {
    if ! command -v nvidia-smi &>/dev/null; then
        log_warn "No NVIDIA driver installed"
        return 0
    fi

    local version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)

    log_info "Uninstalling NVIDIA driver version: $version"

    # Unload modules
    log_info "Unloading kernel modules..."
    rmmod nvidia_drm nvidia_modeset nvidia_uvm nvidia 2>/dev/null || true

    # Remove module files
    local kernel=$(uname -r)
    log_info "Removing module files..."
    rm -f "/lib/modules/$kernel/kernel/drivers/video"/nvidia*.ko

    # Update module dependencies
    depmod -a

    # Save state
    save_state "$version" "uninstall"

    log_success "Driver uninstalled successfully"
}

# Rollback to previous version
rollback_driver() {
    if [ ! -f "$STATE_FILE" ]; then
        log_error "No installation history found"
        return 1
    fi

    # Get previous version (second to last entry)
    local prev_version=$(jq -r '.history[-2].version // empty' "$STATE_FILE")

    if [ -z "$prev_version" ]; then
        log_error "No previous version found in history"
        return 1
    fi

    log_info "Rolling back to previous version: $prev_version"
    install_driver "$prev_version" true true
}

# Verify installation
verify_installation() {
    log_info "Verifying NVIDIA driver installation..."
    echo ""

    # Check nvidia-smi
    if ! command -v nvidia-smi &>/dev/null; then
        log_error "nvidia-smi not found"
        return 1
    fi

    # Run nvidia-smi
    if ! nvidia-smi; then
        log_error "nvidia-smi failed"
        return 1
    fi

    echo ""

    # Check modules
    log_info "Checking kernel modules..."
    local modules=("nvidia" "nvidia_uvm" "nvidia_modeset" "nvidia_drm")
    local missing=0

    for mod in "${modules[@]}"; do
        if lsmod | grep -q "^$mod "; then
            log_success "Module loaded: $mod"
        else
            log_warn "Module not loaded: $mod"
            missing=$((missing + 1))
        fi
    done

    echo ""

    # Check device nodes
    log_info "Checking device nodes..."
    if [ -e "/dev/nvidia0" ]; then
        log_success "Device nodes present"
        ls -la /dev/nvidia* | head -5
    else
        log_error "Device nodes not found"
        return 1
    fi

    echo ""

    if [ $missing -eq 0 ]; then
        log_success "✓ Driver verification passed"
        return 0
    else
        log_warn "⚠ Verification completed with warnings"
        return 1
    fi
}

# Clean old packages
clean_packages() {
    log_info "Cleaning old driver packages..."

    local current_kernel=$(uname -r)
    local removed=0
    local size_freed=0

    # Find packages for other kernels
    while IFS= read -r package; do
        if [[ ! "$package" =~ $current_kernel ]]; then
            local size=$(stat -c%s "$package")
            size_freed=$((size_freed + size))

            log_info "Removing: $(basename $package)"
            rm -f "$package" "${package}.sha256"
            removed=$((removed + 1))
        fi
    done < <(find "$REPO_DIR" -name "nvidia-driver-*.tar.gz" -type f)

    if [ $removed -gt 0 ]; then
        local size_mb=$((size_freed / 1024 / 1024))
        log_success "Removed $removed packages, freed ${size_mb} MB"
    else
        log_info "No packages to clean"
    fi
}

# Show driver details
show_driver() {
    local version=$1
    local package=$(find_driver_package "$version")

    if [ -z "$package" ]; then
        log_error "Driver package not found: $version"
        return 1
    fi

    log_info "Driver Package Details:"
    echo ""

    # Basic info
    printf "%-20s: %s\n" "Package" "$(basename $package)"
    printf "%-20s: %s\n" "Location" "$package"
    printf "%-20s: %s\n" "Size" "$(du -h $package | cut -f1)"
    printf "%-20s: %s\n" "Modified" "$(stat -c %y $package | cut -d' ' -f1,2)"

    # Checksum
    if [ -f "${package}.sha256" ]; then
        printf "%-20s: %s\n" "SHA256" "$(cat ${package}.sha256 | cut -d' ' -f1)"
    fi

    echo ""

    # Extract and show metadata
    local temp_dir=$(mktemp -d)
    tar -xzf "$package" -C "$temp_dir"

    if [ -f "$temp_dir"/*/metadata.json ]; then
        log_info "Metadata:"
        cat "$temp_dir"/*/metadata.json | jq '.'
    fi

    echo ""

    # Show contents
    log_info "Package Contents:"
    tar -tzf "$package" | head -20

    # Cleanup
    rm -rf "$temp_dir"
}

# Download driver
download_driver() {
    local version=$1
    local package=$(find_driver_package "$version")

    if [ -z "$package" ]; then
        log_error "Driver package not found: $version"
        return 1
    fi

    local dest="./$(basename $package)"

    log_info "Downloading: $(basename $package)"
    cp "$package" "$dest"

    if [ -f "${package}.sha256" ]; then
        cp "${package}.sha256" "${dest}.sha256"
    fi

    log_success "Downloaded to: $dest"
}

# Search for drivers
search_drivers() {
    local kernel=$1

    log_info "Searching drivers for kernel: $kernel"
    echo ""

    local packages=$(find "$REPO_DIR" -name "*-kernel-${kernel}*.tar.gz" -type f | sort -V)

    if [ -z "$packages" ]; then
        log_warn "No drivers found for kernel: $kernel"
        return 0
    fi

    printf "%-20s %-30s %-10s\n" "DRIVER VERSION" "KERNEL VERSION" "SIZE"
    printf "%s\n" "------------------------------------------------------------"

    while IFS= read -r package; do
        local basename=$(basename "$package" .tar.gz)

        if [[ $basename =~ nvidia-driver-([0-9.]+)-kernel-([^ ]+) ]]; then
            local driver_ver="${BASH_REMATCH[1]}"
            local kernel_ver="${BASH_REMATCH[2]}"
            local size=$(du -h "$package" | cut -f1)

            printf "%-20s %-30s %-10s\n" "$driver_ver" "$kernel_ver" "$size"
        fi
    done <<< "$packages"
}

# Update repository index
update_index() {
    log_info "Updating driver repository index..."

    local index_file="${REPO_DIR}/index.json"
    local temp_index=$(mktemp)

    cat > "$temp_index" << EOF
{
  "repository": "NVIDIA Precompiled Drivers",
  "last_updated": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "drivers": [
EOF

    local first=true

    while IFS= read -r package; do
        local basename=$(basename "$package" .tar.gz)

        if [[ $basename =~ nvidia-driver-([0-9.]+)-kernel-([^ ]+) ]]; then
            local driver_ver="${BASH_REMATCH[1]}"
            local kernel_ver="${BASH_REMATCH[2]}"
            local size=$(stat -c%s "$package")
            local sha256=$(sha256sum "$package" | cut -d' ' -f1)

            if [ "$first" = false ]; then
                echo "," >> "$temp_index"
            fi
            first=false

            cat >> "$temp_index" << EOF
    {
      "driver_version": "$driver_ver",
      "kernel_version": "$kernel_ver",
      "filename": "$(basename $package)",
      "size": $size,
      "sha256": "$sha256",
      "path": "$package"
    }
EOF
        fi
    done < <(find "$REPO_DIR" -name "nvidia-driver-*.tar.gz" -type f | sort -V)

    cat >> "$temp_index" << EOF

  ]
}
EOF

    mv "$temp_index" "$index_file"
    log_success "Index updated: $index_file"
}

# Main command dispatcher
main() {
    init_environment

    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi

    local command=$1
    shift

    case "$command" in
        list)
            list_drivers
            ;;
        list-installed)
            list_installed
            ;;
        install)
            if [ $# -lt 1 ]; then
                log_error "Usage: $0 install <version>"
                exit 1
            fi
            install_driver "$1"
            ;;
        install-latest)
            install_latest
            ;;
        uninstall)
            uninstall_driver
            ;;
        rollback)
            rollback_driver
            ;;
        verify)
            verify_installation
            ;;
        clean)
            clean_packages
            ;;
        show)
            if [ $# -lt 1 ]; then
                log_error "Usage: $0 show <version>"
                exit 1
            fi
            show_driver "$1"
            ;;
        download)
            if [ $# -lt 1 ]; then
                log_error "Usage: $0 download <version>"
                exit 1
            fi
            download_driver "$1"
            ;;
        search)
            if [ $# -lt 1 ]; then
                log_error "Usage: $0 search <kernel>"
                exit 1
            fi
            search_drivers "$1"
            ;;
        update-index)
            update_index
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Run main
main "$@"

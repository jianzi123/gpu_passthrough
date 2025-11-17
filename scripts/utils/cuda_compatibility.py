#!/usr/bin/env python3
"""
CUDA Compatibility Matrix for Different GPU Models
Provides mapping between GPU models and supported CUDA versions
"""

# GPU Model to CUDA Version Compatibility Matrix
GPU_CUDA_COMPATIBILITY = {
    # NVIDIA V100 (Volta, Compute Capability 7.0)
    "Tesla V100": {
        "compute_capability": "7.0",
        "architecture": "Volta",
        "min_cuda_version": "9.0",
        "recommended_cuda_version": "12.2",
        "max_cuda_version": "12.4",
        "min_driver_version": "396.26",
        "recommended_driver_version": "535.154.05",
        "supported_cuda_versions": ["11.8", "12.0", "12.1", "12.2", "12.3", "12.4"]
    },
    "V100-SXM2": {
        "compute_capability": "7.0",
        "architecture": "Volta",
        "min_cuda_version": "9.0",
        "recommended_cuda_version": "12.2",
        "max_cuda_version": "12.4",
        "min_driver_version": "396.26",
        "recommended_driver_version": "535.154.05",
        "supported_cuda_versions": ["11.8", "12.0", "12.1", "12.2", "12.3", "12.4"]
    },
    "V100-PCIE": {
        "compute_capability": "7.0",
        "architecture": "Volta",
        "min_cuda_version": "9.0",
        "recommended_cuda_version": "12.2",
        "max_cuda_version": "12.4",
        "min_driver_version": "396.26",
        "recommended_driver_version": "535.154.05",
        "supported_cuda_versions": ["11.8", "12.0", "12.1", "12.2", "12.3", "12.4"]
    },

    # NVIDIA A100 (Ampere, Compute Capability 8.0)
    "A100-SXM4": {
        "compute_capability": "8.0",
        "architecture": "Ampere",
        "min_cuda_version": "11.0",
        "recommended_cuda_version": "12.2",
        "max_cuda_version": "12.4",
        "min_driver_version": "450.51.06",
        "recommended_driver_version": "535.154.05",
        "supported_cuda_versions": ["11.8", "12.0", "12.1", "12.2", "12.3", "12.4"]
    },
    "A100-PCIE": {
        "compute_capability": "8.0",
        "architecture": "Ampere",
        "min_cuda_version": "11.0",
        "recommended_cuda_version": "12.2",
        "max_cuda_version": "12.4",
        "min_driver_version": "450.51.06",
        "recommended_driver_version": "535.154.05",
        "supported_cuda_versions": ["11.8", "12.0", "12.1", "12.2", "12.3", "12.4"]
    },
    "Tesla A100": {
        "compute_capability": "8.0",
        "architecture": "Ampere",
        "min_cuda_version": "11.0",
        "recommended_cuda_version": "12.2",
        "max_cuda_version": "12.4",
        "min_driver_version": "450.51.06",
        "recommended_driver_version": "535.154.05",
        "supported_cuda_versions": ["11.8", "12.0", "12.1", "12.2", "12.3", "12.4"]
    },

    # NVIDIA H100 (Hopper, Compute Capability 9.0)
    "H100": {
        "compute_capability": "9.0",
        "architecture": "Hopper",
        "min_cuda_version": "11.8",
        "recommended_cuda_version": "12.3",
        "max_cuda_version": "12.4",
        "min_driver_version": "520.61.05",
        "recommended_driver_version": "545.23.08",
        "supported_cuda_versions": ["11.8", "12.0", "12.1", "12.2", "12.3", "12.4"],
        "notes": "H100 requires CUDA 11.8+ for full feature support including FP8"
    },
    "H100-SXM5": {
        "compute_capability": "9.0",
        "architecture": "Hopper",
        "min_cuda_version": "11.8",
        "recommended_cuda_version": "12.3",
        "max_cuda_version": "12.4",
        "min_driver_version": "520.61.05",
        "recommended_driver_version": "545.23.08",
        "supported_cuda_versions": ["11.8", "12.0", "12.1", "12.2", "12.3", "12.4"]
    },
    "H100-PCIE": {
        "compute_capability": "9.0",
        "architecture": "Hopper",
        "min_cuda_version": "11.8",
        "recommended_cuda_version": "12.3",
        "max_cuda_version": "12.4",
        "min_driver_version": "520.61.05",
        "recommended_driver_version": "545.23.08",
        "supported_cuda_versions": ["11.8", "12.0", "12.1", "12.2", "12.3", "12.4"]
    },

    # NVIDIA RTX 4090 (Ada Lovelace, Compute Capability 8.9)
    "RTX 4090": {
        "compute_capability": "8.9",
        "architecture": "Ada Lovelace",
        "min_cuda_version": "11.8",
        "recommended_cuda_version": "12.2",
        "max_cuda_version": "12.4",
        "min_driver_version": "520.61.05",
        "recommended_driver_version": "535.154.05",
        "supported_cuda_versions": ["11.8", "12.0", "12.1", "12.2", "12.3", "12.4"]
    },
    "GeForce RTX 4090": {
        "compute_capability": "8.9",
        "architecture": "Ada Lovelace",
        "min_cuda_version": "11.8",
        "recommended_cuda_version": "12.2",
        "max_cuda_version": "12.4",
        "min_driver_version": "520.61.05",
        "recommended_driver_version": "535.154.05",
        "supported_cuda_versions": ["11.8", "12.0", "12.1", "12.2", "12.3", "12.4"]
    },

    # NVIDIA RTX 3090 (Ampere, Compute Capability 8.6)
    "RTX 3090": {
        "compute_capability": "8.6",
        "architecture": "Ampere",
        "min_cuda_version": "11.1",
        "recommended_cuda_version": "12.2",
        "max_cuda_version": "12.4",
        "min_driver_version": "455.32.00",
        "recommended_driver_version": "535.154.05",
        "supported_cuda_versions": ["11.8", "12.0", "12.1", "12.2", "12.3", "12.4"]
    },
    "GeForce RTX 3090": {
        "compute_capability": "8.6",
        "architecture": "Ampere",
        "min_cuda_version": "11.1",
        "recommended_cuda_version": "12.2",
        "max_cuda_version": "12.4",
        "min_driver_version": "455.32.00",
        "recommended_driver_version": "535.154.05",
        "supported_cuda_versions": ["11.8", "12.0", "12.1", "12.2", "12.3", "12.4"]
    },

    # NVIDIA A800 (Ampere, Compute Capability 8.0)
    "A800": {
        "compute_capability": "8.0",
        "architecture": "Ampere",
        "min_cuda_version": "11.0",
        "recommended_cuda_version": "12.2",
        "max_cuda_version": "12.4",
        "min_driver_version": "450.51.06",
        "recommended_driver_version": "535.154.05",
        "supported_cuda_versions": ["11.8", "12.0", "12.1", "12.2", "12.3", "12.4"]
    },

    # NVIDIA H800 (Hopper, Compute Capability 9.0)
    "H800": {
        "compute_capability": "9.0",
        "architecture": "Hopper",
        "min_cuda_version": "11.8",
        "recommended_cuda_version": "12.3",
        "max_cuda_version": "12.4",
        "min_driver_version": "520.61.05",
        "recommended_driver_version": "545.23.08",
        "supported_cuda_versions": ["11.8", "12.0", "12.1", "12.2", "12.3", "12.4"]
    },
}

# CUDA Version to Driver Version Compatibility
CUDA_DRIVER_COMPATIBILITY = {
    "12.4": {
        "min_driver_linux": "550.54.15",
        "recommended_driver_linux": "550.90.07"
    },
    "12.3": {
        "min_driver_linux": "545.23.06",
        "recommended_driver_linux": "545.23.08"
    },
    "12.2": {
        "min_driver_linux": "535.54.03",
        "recommended_driver_linux": "535.154.05"
    },
    "12.1": {
        "min_driver_linux": "530.30.02",
        "recommended_driver_linux": "530.30.02"
    },
    "12.0": {
        "min_driver_linux": "525.60.13",
        "recommended_driver_linux": "525.125.06"
    },
    "11.8": {
        "min_driver_linux": "520.61.05",
        "recommended_driver_linux": "520.61.05"
    },
}


def get_gpu_cuda_info(gpu_name: str) -> dict:
    """
    Get CUDA compatibility information for a GPU model

    Args:
        gpu_name: GPU model name (e.g., "A100-SXM4", "H100", "RTX 4090")

    Returns:
        Dictionary containing CUDA compatibility info
    """
    # Try exact match first
    if gpu_name in GPU_CUDA_COMPATIBILITY:
        return GPU_CUDA_COMPATIBILITY[gpu_name]

    # Try partial match (case insensitive)
    gpu_name_lower = gpu_name.lower()
    for key, value in GPU_CUDA_COMPATIBILITY.items():
        if key.lower() in gpu_name_lower or gpu_name_lower in key.lower():
            return value

    return None


def get_recommended_cuda_version(gpu_name: str) -> str:
    """
    Get recommended CUDA version for a GPU model

    Args:
        gpu_name: GPU model name

    Returns:
        Recommended CUDA version string (e.g., "12.2")
    """
    info = get_gpu_cuda_info(gpu_name)
    if info:
        return info.get("recommended_cuda_version")
    return None


def get_recommended_driver_version(gpu_name: str = None, cuda_version: str = None) -> str:
    """
    Get recommended driver version for GPU or CUDA version

    Args:
        gpu_name: GPU model name (optional)
        cuda_version: CUDA version (optional)

    Returns:
        Recommended driver version string
    """
    if gpu_name:
        info = get_gpu_cuda_info(gpu_name)
        if info:
            return info.get("recommended_driver_version")

    if cuda_version:
        driver_info = CUDA_DRIVER_COMPATIBILITY.get(cuda_version)
        if driver_info:
            return driver_info.get("recommended_driver_linux")

    return None


def is_cuda_compatible(gpu_name: str, cuda_version: str) -> bool:
    """
    Check if a CUDA version is compatible with a GPU model

    Args:
        gpu_name: GPU model name
        cuda_version: CUDA version to check

    Returns:
        True if compatible, False otherwise
    """
    info = get_gpu_cuda_info(gpu_name)
    if not info:
        return False

    supported_versions = info.get("supported_cuda_versions", [])
    return cuda_version in supported_versions


def print_compatibility_matrix():
    """Print the complete GPU-CUDA compatibility matrix"""
    print("\n=== GPU to CUDA Version Compatibility Matrix ===\n")

    architectures = {}
    for gpu, info in GPU_CUDA_COMPATIBILITY.items():
        arch = info["architecture"]
        if arch not in architectures:
            architectures[arch] = []
        architectures[arch].append((gpu, info))

    for arch in ["Volta", "Ampere", "Hopper", "Ada Lovelace"]:
        if arch not in architectures:
            continue

        print(f"\n{arch} Architecture:")
        print("-" * 80)

        for gpu, info in architectures[arch]:
            print(f"\nGPU Model: {gpu}")
            print(f"  Compute Capability: {info['compute_capability']}")
            print(f"  Min CUDA Version: {info['min_cuda_version']}")
            print(f"  Recommended CUDA: {info['recommended_cuda_version']}")
            print(f"  Max CUDA Version: {info['max_cuda_version']}")
            print(f"  Min Driver: {info['min_driver_version']}")
            print(f"  Recommended Driver: {info['recommended_driver_version']}")
            print(f"  Supported CUDA: {', '.join(info['supported_cuda_versions'])}")
            if "notes" in info:
                print(f"  Notes: {info['notes']}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--matrix":
            print_compatibility_matrix()
        else:
            gpu_name = sys.argv[1]
            info = get_gpu_cuda_info(gpu_name)
            if info:
                print(f"\nGPU: {gpu_name}")
                print(f"Architecture: {info['architecture']}")
                print(f"Compute Capability: {info['compute_capability']}")
                print(f"Recommended CUDA: {info['recommended_cuda_version']}")
                print(f"Recommended Driver: {info['recommended_driver_version']}")
                print(f"Supported CUDA Versions: {', '.join(info['supported_cuda_versions'])}")
            else:
                print(f"GPU model '{gpu_name}' not found in compatibility matrix")
                sys.exit(1)
    else:
        print_compatibility_matrix()

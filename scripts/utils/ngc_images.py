#!/usr/bin/env python3
"""
NGC (NVIDIA GPU Cloud) Container Image Registry
Provides configuration and management for NGC container images
"""

# NGC Container Image Registry
NGC_IMAGES = {
    "pytorch": {
        "name": "PyTorch",
        "description": "NVIDIA optimized PyTorch container with CUDA, cuDNN, NCCL",
        "registry": "nvcr.io/nvidia/pytorch",
        "versions": {
            "24.01": {
                "tag": "24.01-py3",
                "pytorch_version": "2.3.0a0",
                "cuda_version": "12.3",
                "python_version": "3.10",
                "ubuntu_version": "22.04",
                "features": ["CUDA 12.3", "cuDNN 8.9", "NCCL 2.19", "TensorRT 8.6"]
            },
            "23.12": {
                "tag": "23.12-py3",
                "pytorch_version": "2.2.0a0",
                "cuda_version": "12.3",
                "python_version": "3.10",
                "ubuntu_version": "22.04",
                "features": ["CUDA 12.3", "cuDNN 8.9", "NCCL 2.19", "TensorRT 8.6"]
            },
            "23.10": {
                "tag": "23.10-py3",
                "pytorch_version": "2.1.0a0",
                "cuda_version": "12.2",
                "python_version": "3.10",
                "ubuntu_version": "22.04",
                "features": ["CUDA 12.2", "cuDNN 8.9", "NCCL 2.18", "TensorRT 8.6"]
            },
            "23.08": {
                "tag": "23.08-py3",
                "pytorch_version": "2.1.0a0",
                "cuda_version": "12.2",
                "python_version": "3.10",
                "ubuntu_version": "22.04",
                "features": ["CUDA 12.2", "cuDNN 8.9", "NCCL 2.18"]
            }
        },
        "default_version": "24.01",
        "use_cases": ["Training", "Inference", "Development"]
    },

    "tensorflow": {
        "name": "TensorFlow",
        "description": "NVIDIA optimized TensorFlow container",
        "registry": "nvcr.io/nvidia/tensorflow",
        "versions": {
            "24.01": {
                "tag": "24.01-tf2-py3",
                "tensorflow_version": "2.15.0",
                "cuda_version": "12.3",
                "python_version": "3.10",
                "ubuntu_version": "22.04",
                "features": ["CUDA 12.3", "cuDNN 8.9", "NCCL 2.19", "TensorRT 8.6"]
            },
            "23.12": {
                "tag": "23.12-tf2-py3",
                "tensorflow_version": "2.14.0",
                "cuda_version": "12.3",
                "python_version": "3.10",
                "ubuntu_version": "22.04",
                "features": ["CUDA 12.3", "cuDNN 8.9", "NCCL 2.19"]
            }
        },
        "default_version": "24.01",
        "use_cases": ["Training", "Inference"]
    },

    "nemo": {
        "name": "NVIDIA NeMo (includes Megatron-LM)",
        "description": "NeMo framework for conversational AI with Megatron-LM",
        "registry": "nvcr.io/nvidia/nemo",
        "versions": {
            "24.01": {
                "tag": "24.01",
                "nemo_version": "1.22.0",
                "megatron_version": "core_0.5.0",
                "pytorch_version": "2.2.0",
                "cuda_version": "12.3",
                "python_version": "3.10",
                "ubuntu_version": "22.04",
                "features": [
                    "Megatron-LM Core 0.5.0",
                    "CUDA 12.3",
                    "cuDNN 8.9",
                    "NCCL 2.19",
                    "Transformer Engine",
                    "Apex"
                ]
            },
            "23.11": {
                "tag": "23.11",
                "nemo_version": "1.21.0",
                "megatron_version": "core_0.4.0",
                "pytorch_version": "2.1.0",
                "cuda_version": "12.2",
                "python_version": "3.10",
                "ubuntu_version": "22.04",
                "features": [
                    "Megatron-LM Core 0.4.0",
                    "CUDA 12.2",
                    "NCCL 2.18",
                    "Transformer Engine"
                ]
            }
        },
        "default_version": "24.01",
        "use_cases": ["LLM Training", "LLM Fine-tuning", "ASR", "TTS", "NLP"]
    },

    "triton": {
        "name": "Triton Inference Server",
        "description": "NVIDIA Triton Inference Server for model deployment",
        "registry": "nvcr.io/nvidia/tritonserver",
        "versions": {
            "24.01": {
                "tag": "24.01-py3",
                "triton_version": "2.42.0",
                "cuda_version": "12.3",
                "python_version": "3.10",
                "ubuntu_version": "22.04",
                "features": [
                    "CUDA 12.3",
                    "TensorRT 8.6",
                    "PyTorch Backend",
                    "TensorFlow Backend",
                    "ONNX Runtime",
                    "Python Backend"
                ],
                "backends": ["TensorRT", "PyTorch", "TensorFlow", "ONNX", "Python"]
            },
            "23.12": {
                "tag": "23.12-py3",
                "triton_version": "2.41.0",
                "cuda_version": "12.3",
                "python_version": "3.10",
                "ubuntu_version": "22.04",
                "features": ["CUDA 12.3", "TensorRT 8.6", "Multiple Backends"]
            }
        },
        "default_version": "24.01",
        "use_cases": ["Inference", "Production Deployment", "Model Serving"]
    },

    "tensorrt": {
        "name": "TensorRT",
        "description": "NVIDIA TensorRT for high-performance inference",
        "registry": "nvcr.io/nvidia/tensorrt",
        "versions": {
            "24.01": {
                "tag": "24.01-py3",
                "tensorrt_version": "8.6.3",
                "cuda_version": "12.3",
                "python_version": "3.10",
                "ubuntu_version": "22.04",
                "features": ["CUDA 12.3", "cuDNN 8.9", "ONNX Parser", "Python API"]
            },
            "23.12": {
                "tag": "23.12-py3",
                "tensorrt_version": "8.6.1",
                "cuda_version": "12.3",
                "python_version": "3.10",
                "ubuntu_version": "22.04",
                "features": ["CUDA 12.3", "cuDNN 8.9"]
            }
        },
        "default_version": "24.01",
        "use_cases": ["Inference Optimization", "Model Conversion"]
    },

    "cuda": {
        "name": "CUDA Development",
        "description": "NVIDIA CUDA development container",
        "registry": "nvcr.io/nvidia/cuda",
        "versions": {
            "12.3.2": {
                "tag": "12.3.2-devel-ubuntu22.04",
                "cuda_version": "12.3.2",
                "ubuntu_version": "22.04",
                "image_type": "devel",
                "features": ["CUDA Toolkit", "NVCC", "cuBLAS", "cuFFT", "cuSPARSE"]
            },
            "12.2.2": {
                "tag": "12.2.2-devel-ubuntu22.04",
                "cuda_version": "12.2.2",
                "ubuntu_version": "22.04",
                "image_type": "devel",
                "features": ["CUDA Toolkit", "Development Tools"]
            },
            "11.8.0": {
                "tag": "11.8.0-devel-ubuntu22.04",
                "cuda_version": "11.8.0",
                "ubuntu_version": "22.04",
                "image_type": "devel",
                "features": ["CUDA Toolkit 11.8", "Legacy Support"]
            }
        },
        "default_version": "12.3.2",
        "use_cases": ["CUDA Development", "Custom Applications"]
    },

    "deepstream": {
        "name": "DeepStream SDK",
        "description": "NVIDIA DeepStream for video analytics and AI",
        "registry": "nvcr.io/nvidia/deepstream",
        "versions": {
            "6.4": {
                "tag": "6.4-triton-multiarch",
                "deepstream_version": "6.4",
                "cuda_version": "12.2",
                "ubuntu_version": "22.04",
                "features": [
                    "Triton Integration",
                    "Video Analytics",
                    "TensorRT",
                    "Multi-stream Support"
                ]
            },
            "6.3": {
                "tag": "6.3-triton",
                "deepstream_version": "6.3",
                "cuda_version": "12.1",
                "ubuntu_version": "22.04",
                "features": ["Triton", "Video Analytics"]
            }
        },
        "default_version": "6.4",
        "use_cases": ["Video Analytics", "Streaming AI"]
    },

    "rapids": {
        "name": "RAPIDS",
        "description": "NVIDIA RAPIDS for GPU-accelerated data science",
        "registry": "nvcr.io/nvidia/rapidsai/rapids",
        "versions": {
            "24.02": {
                "tag": "24.02-cuda12.0-py3.10",
                "rapids_version": "24.02",
                "cuda_version": "12.0",
                "python_version": "3.10",
                "features": ["cuDF", "cuML", "cuGraph", "cuSpatial", "Dask"]
            },
            "23.12": {
                "tag": "23.12-cuda12.0-py3.10",
                "rapids_version": "23.12",
                "cuda_version": "12.0",
                "python_version": "3.10",
                "features": ["cuDF", "cuML", "cuGraph"]
            }
        },
        "default_version": "24.02",
        "use_cases": ["Data Science", "ML Preprocessing", "ETL"]
    }
}


def get_ngc_image_info(image_name: str, version: str = None) -> dict:
    """
    Get NGC image information

    Args:
        image_name: NGC image name (e.g., "pytorch", "triton", "nemo")
        version: Specific version (optional, uses default if not specified)

    Returns:
        Dictionary containing image information
    """
    if image_name not in NGC_IMAGES:
        return None

    image_info = NGC_IMAGES[image_name].copy()

    if version is None:
        version = image_info["default_version"]

    if version not in image_info["versions"]:
        return None

    version_info = image_info["versions"][version]
    image_info["selected_version"] = version
    image_info["version_info"] = version_info

    return image_info


def get_image_url(image_name: str, version: str = None) -> str:
    """
    Get full NGC image URL

    Args:
        image_name: NGC image name
        version: Specific version (optional)

    Returns:
        Full image URL (e.g., "nvcr.io/nvidia/pytorch:24.01-py3")
    """
    info = get_ngc_image_info(image_name, version)
    if not info:
        return None

    registry = info["registry"]
    tag = info["version_info"]["tag"]

    return f"{registry}:{tag}"


def get_images_by_cuda_version(cuda_version: str) -> list:
    """
    Get all NGC images compatible with a CUDA version

    Args:
        cuda_version: CUDA version (e.g., "12.3", "12.2")

    Returns:
        List of compatible image names and versions
    """
    compatible_images = []

    for image_name, image_data in NGC_IMAGES.items():
        for version, version_info in image_data["versions"].items():
            if version_info.get("cuda_version", "").startswith(cuda_version):
                compatible_images.append({
                    "image": image_name,
                    "version": version,
                    "name": image_data["name"],
                    "url": get_image_url(image_name, version)
                })

    return compatible_images


def print_ngc_catalog():
    """Print NGC image catalog"""
    print("\n=== NGC Container Image Catalog ===\n")

    for image_name, image_data in NGC_IMAGES.items():
        print(f"\n{image_data['name']} ({image_name})")
        print("-" * 80)
        print(f"Description: {image_data['description']}")
        print(f"Registry: {image_data['registry']}")
        print(f"Use Cases: {', '.join(image_data['use_cases'])}")
        print(f"Default Version: {image_data['default_version']}")

        print("\nAvailable Versions:")
        for version, version_info in image_data["versions"].items():
            tag = version_info["tag"]
            cuda = version_info.get("cuda_version", "N/A")
            print(f"  {version} (CUDA {cuda})")
            print(f"    Tag: {tag}")
            print(f"    URL: {image_data['registry']}:{tag}")
            if "features" in version_info:
                print(f"    Features: {', '.join(version_info['features'])}")


def get_pull_command(image_name: str, version: str = None) -> str:
    """
    Get docker pull command for an NGC image

    Args:
        image_name: NGC image name
        version: Specific version (optional)

    Returns:
        Docker pull command string
    """
    url = get_image_url(image_name, version)
    if url:
        return f"docker pull {url}"
    return None


def get_run_command(image_name: str, version: str = None, gpus: str = "all",
                    interactive: bool = True, volumes: list = None) -> str:
    """
    Get docker run command for an NGC image

    Args:
        image_name: NGC image name
        version: Specific version (optional)
        gpus: GPU specification (default: "all")
        interactive: Run in interactive mode (default: True)
        volumes: List of volume mounts

    Returns:
        Docker run command string
    """
    url = get_image_url(image_name, version)
    if not url:
        return None

    cmd_parts = ["docker run"]

    # GPU support
    cmd_parts.append(f"--gpus {gpus}")

    # Interactive mode
    if interactive:
        cmd_parts.append("-it")

    # Remove container on exit
    cmd_parts.append("--rm")

    # Network
    cmd_parts.append("--network=host")

    # IPC mode for shared memory
    cmd_parts.append("--ipc=host")

    # Volumes
    if volumes:
        for vol in volumes:
            cmd_parts.append(f"-v {vol}")

    # Image
    cmd_parts.append(url)

    return " ".join(cmd_parts)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--catalog":
            print_ngc_catalog()
        elif sys.argv[1] == "--pull":
            if len(sys.argv) > 2:
                image_name = sys.argv[2]
                version = sys.argv[3] if len(sys.argv) > 3 else None
                cmd = get_pull_command(image_name, version)
                if cmd:
                    print(cmd)
                else:
                    print(f"Image '{image_name}' not found")
                    sys.exit(1)
        elif sys.argv[1] == "--run":
            if len(sys.argv) > 2:
                image_name = sys.argv[2]
                version = sys.argv[3] if len(sys.argv) > 3 else None
                cmd = get_run_command(image_name, version)
                if cmd:
                    print(cmd)
                else:
                    print(f"Image '{image_name}' not found")
                    sys.exit(1)
        elif sys.argv[1] == "--cuda":
            if len(sys.argv) > 2:
                cuda_version = sys.argv[2]
                images = get_images_by_cuda_version(cuda_version)
                print(f"\nNGC Images compatible with CUDA {cuda_version}:\n")
                for img in images:
                    print(f"  {img['name']}: {img['url']}")
        else:
            image_name = sys.argv[1]
            version = sys.argv[2] if len(sys.argv) > 2 else None
            info = get_ngc_image_info(image_name, version)
            if info:
                print(f"\nImage: {info['name']}")
                print(f"Description: {info['description']}")
                print(f"Version: {info['selected_version']}")
                print(f"URL: {get_image_url(image_name, version)}")
                print(f"\nVersion Details:")
                for key, value in info['version_info'].items():
                    print(f"  {key}: {value}")
            else:
                print(f"Image '{image_name}' version '{version}' not found")
                sys.exit(1)
    else:
        print_ngc_catalog()

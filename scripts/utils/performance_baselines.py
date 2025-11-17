#!/usr/bin/env python3
"""
GPU and CPU Performance Baseline Database
Contains expected performance metrics for different GPU/CPU models
Based on official specifications and real-world benchmarks
"""

import json
from typing import Dict, Any

# GPU Performance Baselines
GPU_BASELINES = {
    # NVIDIA Ampere Architecture
    "A100-SXM4-40GB": {
        "architecture": "Ampere",
        "compute_capability": "8.0",
        "memory_gb": 40,
        "memory_bandwidth_gbs": 1555,
        "fp64_tflops": 9.7,
        "fp32_tflops": 19.5,
        "tf32_tflops": 156,
        "fp16_tflops": 312,
        "int8_tops": 624,
        "nvlink_version": "3.0",
        "nvlink_bandwidth_gbs": 600,  # 12 links x 50GB/s
        "pcie_gen": 4,
        "pcie_bandwidth_gbs": 64,  # PCIe Gen4 x16
        "tdp_watts": 400,
        "expected_mfu": {  # Model FLOP Utilization
            "megatron_gpt": 0.52,  # 52% on large models
            "bert_large": 0.45,
        },
        "nccl_allreduce_busbw_gbs": {
            "intra_node_8gpu": 250,  # Expected bus bandwidth
            "inter_node_ib_hdr": 180,  # InfiniBand HDR 200Gbps
        }
    },
    "A100-SXM4-80GB": {
        "architecture": "Ampere",
        "compute_capability": "8.0",
        "memory_gb": 80,
        "memory_bandwidth_gbs": 2039,
        "fp64_tflops": 9.7,
        "fp32_tflops": 19.5,
        "tf32_tflops": 156,
        "fp16_tflops": 312,
        "int8_tops": 624,
        "nvlink_version": "3.0",
        "nvlink_bandwidth_gbs": 600,
        "pcie_gen": 4,
        "pcie_bandwidth_gbs": 64,
        "tdp_watts": 400,
        "expected_mfu": {
            "megatron_gpt": 0.52,
            "bert_large": 0.45,
        },
        "nccl_allreduce_busbw_gbs": {
            "intra_node_8gpu": 250,
            "inter_node_ib_hdr": 180,
        }
    },
    "A100-PCIE-40GB": {
        "architecture": "Ampere",
        "compute_capability": "8.0",
        "memory_gb": 40,
        "memory_bandwidth_gbs": 1555,
        "fp64_tflops": 9.7,
        "fp32_tflops": 19.5,
        "tf32_tflops": 156,
        "fp16_tflops": 312,
        "int8_tops": 624,
        "nvlink_version": None,
        "nvlink_bandwidth_gbs": 0,
        "pcie_gen": 4,
        "pcie_bandwidth_gbs": 64,
        "tdp_watts": 250,
        "expected_mfu": {
            "megatron_gpt": 0.48,  # Slightly lower due to PCIe
            "bert_large": 0.42,
        },
        "nccl_allreduce_busbw_gbs": {
            "intra_node_8gpu": 50,  # Limited by PCIe
            "inter_node_ib_hdr": 180,
        }
    },

    # NVIDIA Hopper Architecture
    "H100-SXM5-80GB": {
        "architecture": "Hopper",
        "compute_capability": "9.0",
        "memory_gb": 80,
        "memory_bandwidth_gbs": 3350,
        "fp64_tflops": 33.5,
        "fp64_tensor_tflops": 60,
        "fp32_tflops": 67,
        "tf32_tflops": 378,
        "fp16_tflops": 756,
        "fp8_tflops": 1513,
        "int8_tops": 1513,
        "nvlink_version": "4.0",
        "nvlink_bandwidth_gbs": 900,  # 18 links x 50GB/s
        "pcie_gen": 5,
        "pcie_bandwidth_gbs": 128,  # PCIe Gen5 x16
        "tdp_watts": 700,
        "expected_mfu": {
            "megatron_gpt": 0.47,  # 47% on H100 clusters
            "bert_large": 0.50,
        },
        "nccl_allreduce_busbw_gbs": {
            "intra_node_8gpu": 450,  # Higher with NVLink 4.0
            "inter_node_ib_ndr": 350,  # InfiniBand NDR 400Gbps
        }
    },
    "H100-PCIE-80GB": {
        "architecture": "Hopper",
        "compute_capability": "9.0",
        "memory_gb": 80,
        "memory_bandwidth_gbs": 2000,
        "fp64_tflops": 33.5,
        "fp64_tensor_tflops": 60,
        "fp32_tflops": 67,
        "tf32_tflops": 378,
        "fp16_tflops": 756,
        "fp8_tflops": 1513,
        "int8_tops": 1513,
        "nvlink_version": None,
        "nvlink_bandwidth_gbs": 0,
        "pcie_gen": 5,
        "pcie_bandwidth_gbs": 128,
        "tdp_watts": 350,
        "expected_mfu": {
            "megatron_gpt": 0.43,
            "bert_large": 0.46,
        },
        "nccl_allreduce_busbw_gbs": {
            "intra_node_8gpu": 100,
            "inter_node_ib_ndr": 350,
        }
    },

    # NVIDIA Volta Architecture
    "V100-SXM2-16GB": {
        "architecture": "Volta",
        "compute_capability": "7.0",
        "memory_gb": 16,
        "memory_bandwidth_gbs": 900,
        "fp64_tflops": 7.8,
        "fp32_tflops": 15.7,
        "fp16_tflops": 125,
        "nvlink_version": "2.0",
        "nvlink_bandwidth_gbs": 300,  # 6 links x 50GB/s
        "pcie_gen": 3,
        "pcie_bandwidth_gbs": 32,  # PCIe Gen3 x16
        "tdp_watts": 300,
        "expected_mfu": {
            "megatron_gpt": 0.30,
            "bert_large": 0.35,
        },
        "nccl_allreduce_busbw_gbs": {
            "intra_node_8gpu": 180,
            "inter_node_ib_edr": 90,  # InfiniBand EDR 100Gbps
        }
    },
    "V100-SXM2-32GB": {
        "architecture": "Volta",
        "compute_capability": "7.0",
        "memory_gb": 32,
        "memory_bandwidth_gbs": 900,
        "fp64_tflops": 7.8,
        "fp32_tflops": 15.7,
        "fp16_tflops": 125,
        "nvlink_version": "2.0",
        "nvlink_bandwidth_gbs": 300,
        "pcie_gen": 3,
        "pcie_bandwidth_gbs": 32,
        "tdp_watts": 300,
        "expected_mfu": {
            "megatron_gpt": 0.30,
            "bert_large": 0.35,
        },
        "nccl_allreduce_busbw_gbs": {
            "intra_node_8gpu": 180,
            "inter_node_ib_edr": 90,
        }
    },
    "V100-PCIE-16GB": {
        "architecture": "Volta",
        "compute_capability": "7.0",
        "memory_gb": 16,
        "memory_bandwidth_gbs": 900,
        "fp64_tflops": 7.8,
        "fp32_tflops": 15.7,
        "fp16_tflops": 125,
        "nvlink_version": None,
        "nvlink_bandwidth_gbs": 0,
        "pcie_gen": 3,
        "pcie_bandwidth_gbs": 32,
        "tdp_watts": 250,
        "expected_mfu": {
            "megatron_gpt": 0.28,
            "bert_large": 0.32,
        },
        "nccl_allreduce_busbw_gbs": {
            "intra_node_8gpu": 25,
            "inter_node_ib_edr": 90,
        }
    },

    # NVIDIA Ada Lovelace (Consumer/Workstation)
    "RTX-4090": {
        "architecture": "Ada Lovelace",
        "compute_capability": "8.9",
        "memory_gb": 24,
        "memory_bandwidth_gbs": 1008,
        "fp32_tflops": 82.6,
        "fp16_tflops": 165.2,
        "nvlink_version": None,
        "nvlink_bandwidth_gbs": 0,
        "pcie_gen": 4,
        "pcie_bandwidth_gbs": 64,
        "tdp_watts": 450,
        "expected_mfu": {
            "megatron_gpt": 0.35,
            "bert_large": 0.38,
        },
        "nccl_allreduce_busbw_gbs": {
            "intra_node_4gpu": 45,
            "inter_node_10gbe": 9,
        }
    },
}

# InfiniBand/Network Baselines
NETWORK_BASELINES = {
    "IB-EDR": {
        "name": "InfiniBand EDR",
        "bandwidth_gbps": 100,
        "bandwidth_gbs": 12.5,
        "latency_us": 0.7,
        "expected_ib_write_bw_gbs": 11.5,  # ~92% efficiency
    },
    "IB-HDR": {
        "name": "InfiniBand HDR",
        "bandwidth_gbps": 200,
        "bandwidth_gbs": 25,
        "latency_us": 0.6,
        "expected_ib_write_bw_gbs": 23,  # ~92% efficiency
    },
    "IB-NDR": {
        "name": "InfiniBand NDR",
        "bandwidth_gbps": 400,
        "bandwidth_gbs": 50,
        "latency_us": 0.5,
        "expected_ib_write_bw_gbs": 46,  # ~92% efficiency
    },
    "ROCE-V2-100G": {
        "name": "RoCE v2 100GbE",
        "bandwidth_gbps": 100,
        "bandwidth_gbs": 12.5,
        "latency_us": 2.0,
        "expected_ib_write_bw_gbs": 11.0,  # ~88% efficiency
    },
    "ROCE-V2-200G": {
        "name": "RoCE v2 200GbE",
        "bandwidth_gbps": 200,
        "bandwidth_gbs": 25,
        "latency_us": 1.5,
        "expected_ib_write_bw_gbs": 22,  # ~88% efficiency
    },
}

# Megatron-LM Training Baselines
MEGATRON_BASELINES = {
    "GPT-1.2B": {
        "parameters": 1.2e9,
        "v100_single_gpu": {
            "tflops": 39,
            "mfu": 0.30,
            "samples_per_sec": 12,
        },
        "a100_single_gpu": {
            "tflops": 93.6,  # 156 * 0.6
            "mfu": 0.60,
            "samples_per_sec": 28,
        },
        "h100_single_gpu": {
            "tflops": 178,
            "mfu": 0.47,
            "samples_per_sec": 45,
        },
    },
    "GPT-8.3B": {
        "parameters": 8.3e9,
        "v100_512gpu": {
            "petaflops": 15.1,
            "scaling_efficiency": 0.76,
        },
        "a100_512gpu": {
            "petaflops": 35,
            "scaling_efficiency": 0.85,
        },
    },
    "GPT-175B": {
        "parameters": 175e9,
        "a100_1024gpu": {
            "training_time_days": 30,
            "petaflops": 160,
        },
        "h100_1024gpu": {
            "training_time_days": 10,
            "petaflops": 480,
        },
    },
    "GPT-1T": {
        "parameters": 1e12,
        "a100_3072gpu": {
            "petaflops": 502,
            "per_gpu_tflops": 163,
            "mfu": 0.52,
            "scaling_efficiency": 0.98,
        },
    },
}


def get_gpu_baseline(gpu_model: str) -> Dict[str, Any]:
    """Get performance baseline for a GPU model"""
    return GPU_BASELINES.get(gpu_model, {})


def get_network_baseline(network_type: str) -> Dict[str, Any]:
    """Get performance baseline for a network type"""
    return NETWORK_BASELINES.get(network_type, {})


def get_megatron_baseline(model_size: str) -> Dict[str, Any]:
    """Get training baseline for Megatron model"""
    return MEGATRON_BASELINES.get(model_size, {})


def list_available_gpus():
    """List all available GPU models in the database"""
    return list(GPU_BASELINES.keys())


def list_available_networks():
    """List all available network types in the database"""
    return list(NETWORK_BASELINES.keys())


def export_baselines_json(filename: str = "performance_baselines.json"):
    """Export all baselines to JSON file"""
    baselines = {
        "gpu_baselines": GPU_BASELINES,
        "network_baselines": NETWORK_BASELINES,
        "megatron_baselines": MEGATRON_BASELINES,
    }
    with open(filename, 'w') as f:
        json.dump(baselines, f, indent=2)
    print(f"Baselines exported to {filename}")


def compare_gpu_performance(gpu1: str, gpu2: str):
    """Compare performance between two GPU models"""
    baseline1 = get_gpu_baseline(gpu1)
    baseline2 = get_gpu_baseline(gpu2)

    if not baseline1 or not baseline2:
        print(f"One or both GPU models not found")
        return

    print(f"\nPerformance Comparison: {gpu1} vs {gpu2}")
    print("=" * 60)

    metrics = [
        ("FP16 TFLOPS", "fp16_tflops"),
        ("Memory Bandwidth (GB/s)", "memory_bandwidth_gbs"),
        ("NVLink Bandwidth (GB/s)", "nvlink_bandwidth_gbs"),
        ("PCIe Bandwidth (GB/s)", "pcie_bandwidth_gbs"),
    ]

    for name, key in metrics:
        val1 = baseline1.get(key, 0)
        val2 = baseline2.get(key, 0)
        if val2 > 0:
            speedup = val1 / val2
            print(f"{name:30s}: {val1:10.1f} vs {val2:10.1f} ({speedup:.2f}x)")
        else:
            print(f"{name:30s}: {val1:10.1f} vs {val2:10.1f}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "list":
            print("Available GPU Models:")
            for gpu in list_available_gpus():
                print(f"  - {gpu}")
            print("\nAvailable Network Types:")
            for net in list_available_networks():
                print(f"  - {net}")

        elif sys.argv[1] == "export":
            filename = sys.argv[2] if len(sys.argv) > 2 else "performance_baselines.json"
            export_baselines_json(filename)

        elif sys.argv[1] == "compare" and len(sys.argv) >= 4:
            compare_gpu_performance(sys.argv[2], sys.argv[3])

        elif sys.argv[1] == "info" and len(sys.argv) >= 3:
            gpu_model = sys.argv[2]
            baseline = get_gpu_baseline(gpu_model)
            if baseline:
                print(f"\n{gpu_model} Performance Baseline:")
                print("=" * 60)
                for key, value in baseline.items():
                    print(f"{key:30s}: {value}")
            else:
                print(f"GPU model '{gpu_model}' not found")
        else:
            print("Usage:")
            print("  python performance_baselines.py list")
            print("  python performance_baselines.py export [filename]")
            print("  python performance_baselines.py compare <gpu1> <gpu2>")
            print("  python performance_baselines.py info <gpu_model>")
    else:
        print("Available GPU Models:")
        for gpu in list_available_gpus():
            baseline = GPU_BASELINES[gpu]
            print(f"\n{gpu}:")
            print(f"  Architecture: {baseline['architecture']}")
            print(f"  Memory: {baseline['memory_gb']}GB")
            print(f"  FP16 TFLOPS: {baseline.get('fp16_tflops', 'N/A')}")
            print(f"  NVLink BW: {baseline['nvlink_bandwidth_gbs']}GB/s")

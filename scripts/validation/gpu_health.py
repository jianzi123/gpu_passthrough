#!/usr/bin/env python3
"""
GPU Health Check Script
Based on pynvml library for direct GPU monitoring
Implements best practices from NVIDIA DCGM and community tools
"""

import json
import sys
import argparse
from datetime import datetime
from typing import Dict, List, Any

try:
    import pynvml
except ImportError:
    print("Error: pynvml library not installed. Install with: pip install pynvml")
    sys.exit(1)


class GPUHealthChecker:
    """GPU health monitoring using NVIDIA Management Library"""

    def __init__(self):
        """Initialize NVML"""
        try:
            pynvml.nvmlInit()
            self.device_count = pynvml.nvmlDeviceGetCount()
        except pynvml.NVMLError as e:
            print(f"Error initializing NVML: {e}")
            sys.exit(1)

    def __del__(self):
        """Cleanup NVML"""
        try:
            pynvml.nvmlShutdown()
        except:
            pass

    def get_driver_version(self) -> str:
        """Get NVIDIA driver version"""
        try:
            return pynvml.nvmlSystemGetDriverVersion()
        except pynvml.NVMLError:
            return "Unknown"

    def get_cuda_version(self) -> str:
        """Get CUDA version"""
        try:
            version = pynvml.nvmlSystemGetCudaDriverVersion()
            major = version // 1000
            minor = (version % 1000) // 10
            return f"{major}.{minor}"
        except pynvml.NVMLError:
            return "Unknown"

    def check_gpu_info(self, index: int) -> Dict[str, Any]:
        """Get detailed information for a specific GPU"""
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)

        info = {
            "index": index,
            "name": pynvml.nvmlDeviceGetName(handle),
            "uuid": pynvml.nvmlDeviceGetUUID(handle),
        }

        # Memory information
        try:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            info["memory"] = {
                "total": mem_info.total,
                "free": mem_info.free,
                "used": mem_info.used,
                "utilization_percent": round((mem_info.used / mem_info.total) * 100, 2)
            }
        except pynvml.NVMLError:
            info["memory"] = "N/A"

        # Temperature
        try:
            info["temperature"] = {
                "gpu": pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
                "unit": "C"
            }
        except pynvml.NVMLError:
            info["temperature"] = "N/A"

        # Utilization
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            info["utilization"] = {
                "gpu": util.gpu,
                "memory": util.memory
            }
        except pynvml.NVMLError:
            info["utilization"] = "N/A"

        # Power
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
            power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
            info["power"] = {
                "draw": round(power, 2),
                "limit": round(power_limit, 2),
                "unit": "W"
            }
        except pynvml.NVMLError:
            info["power"] = "N/A"

        # PCIe information
        try:
            pcie_gen = pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle)
            pcie_width = pynvml.nvmlDeviceGetCurrPcieLinkWidth(handle)
            max_pcie_gen = pynvml.nvmlDeviceGetMaxPcieLinkGeneration(handle)
            max_pcie_width = pynvml.nvmlDeviceGetMaxPcieLinkWidth(handle)
            info["pcie"] = {
                "current_gen": pcie_gen,
                "current_width": pcie_width,
                "max_gen": max_pcie_gen,
                "max_width": max_pcie_width
            }
        except pynvml.NVMLError:
            info["pcie"] = "N/A"

        # ECC errors
        try:
            ecc_single = pynvml.nvmlDeviceGetTotalEccErrors(
                handle, pynvml.NVML_MEMORY_ERROR_TYPE_CORRECTED, pynvml.NVML_VOLATILE_ECC
            )
            ecc_double = pynvml.nvmlDeviceGetTotalEccErrors(
                handle, pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED, pynvml.NVML_VOLATILE_ECC
            )
            info["ecc_errors"] = {
                "corrected": ecc_single,
                "uncorrected": ecc_double
            }
        except pynvml.NVMLError:
            info["ecc_errors"] = "N/A"

        # Processes
        try:
            processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            info["processes"] = len(processes)
        except pynvml.NVMLError:
            info["processes"] = 0

        return info

    def perform_health_checks(self) -> List[Dict[str, Any]]:
        """Perform health checks on all GPUs"""
        checks = []

        for i in range(self.device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            gpu_checks = []

            # Temperature check
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                if temp < 85:
                    gpu_checks.append({"check": "temperature", "status": "pass", "value": temp})
                elif temp < 95:
                    gpu_checks.append({"check": "temperature", "status": "warn", "value": temp})
                else:
                    gpu_checks.append({"check": "temperature", "status": "fail", "value": temp})
            except pynvml.NVMLError:
                gpu_checks.append({"check": "temperature", "status": "unknown"})

            # ECC errors check
            try:
                ecc_errors = pynvml.nvmlDeviceGetTotalEccErrors(
                    handle, pynvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED, pynvml.NVML_VOLATILE_ECC
                )
                if ecc_errors == 0:
                    gpu_checks.append({"check": "ecc_errors", "status": "pass", "value": 0})
                else:
                    gpu_checks.append({"check": "ecc_errors", "status": "fail", "value": ecc_errors})
            except pynvml.NVMLError:
                gpu_checks.append({"check": "ecc_errors", "status": "n/a"})

            # Power check
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
                power_percent = (power / power_limit) * 100

                if power_percent < 95:
                    gpu_checks.append({"check": "power", "status": "pass", "value": round(power, 2)})
                else:
                    gpu_checks.append({"check": "power", "status": "warn", "value": round(power, 2)})
            except pynvml.NVMLError:
                gpu_checks.append({"check": "power", "status": "unknown"})

            checks.append({
                "gpu_index": i,
                "checks": gpu_checks
            })

        return checks

    def generate_report(self, output_format: str = "json") -> str:
        """Generate health check report"""
        report = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "driver_version": self.get_driver_version(),
            "cuda_version": self.get_cuda_version(),
            "gpu_count": self.device_count,
            "gpus": [],
            "health_checks": []
        }

        # Collect GPU information
        for i in range(self.device_count):
            report["gpus"].append(self.check_gpu_info(i))

        # Perform health checks
        report["health_checks"] = self.perform_health_checks()

        # Determine overall status
        all_checks = []
        for gpu_check in report["health_checks"]:
            for check in gpu_check["checks"]:
                all_checks.append(check.get("status", "unknown"))

        if "fail" in all_checks:
            report["overall_status"] = "fail"
        elif "warn" in all_checks:
            report["overall_status"] = "warn"
        else:
            report["overall_status"] = "pass"

        if output_format == "json":
            return json.dumps(report, indent=2)
        else:
            return str(report)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="GPU Health Check Tool - Based on NVIDIA NVML"
    )
    parser.add_argument(
        "-o", "--output",
        default="/tmp/gpu_health.json",
        help="Output file path (default: /tmp/gpu_health.json)"
    )
    parser.add_argument(
        "-f", "--format",
        default="json",
        choices=["json"],
        help="Output format (default: json)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    # Create health checker
    checker = GPUHealthChecker()

    # Generate report
    report_str = checker.generate_report(args.format)

    # Save to file
    with open(args.output, 'w') as f:
        f.write(report_str)

    # Print to stdout if verbose
    if args.verbose:
        print(report_str)

    # Parse for exit code
    report = json.loads(report_str)
    print(f"\nGPU Health Check Complete")
    print(f"Status: {report['overall_status']}")
    print(f"GPUs: {report['gpu_count']}")
    print(f"Report saved to: {args.output}")

    if report["overall_status"] == "fail":
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()

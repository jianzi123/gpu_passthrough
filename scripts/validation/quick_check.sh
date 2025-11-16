#!/bin/bash
# Quick GPU validation check
# Based on community best practices and NVIDIA tools
# Validation Level 1: 1-5 minutes

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Output file
OUTPUT_FILE="${1:-/tmp/gpu_quick_check_$(date +%Y%m%d_%H%M%S).json}"

echo "Starting GPU Quick Validation..."
echo "Output will be saved to: $OUTPUT_FILE"

# Initialize result
OVERALL_STATUS="pass"
CHECKS=()

# Function to add check result
add_check() {
    local name="$1"
    local status="$2"
    local details="$3"

    CHECKS+=("{\"name\":\"$name\",\"status\":\"$status\",\"details\":\"$details\"}")

    if [ "$status" == "fail" ]; then
        OVERALL_STATUS="fail"
        echo -e "${RED}✗${NC} $name: FAILED - $details"
    else
        echo -e "${GREEN}✓${NC} $name: PASSED"
    fi
}

# Check 1: nvidia-smi availability
echo ""
echo "Check 1: nvidia-smi command availability"
if command -v nvidia-smi &> /dev/null; then
    add_check "nvidia_smi_available" "pass" "nvidia-smi command found"
else
    add_check "nvidia_smi_available" "fail" "nvidia-smi command not found"
    echo -e "${RED}CRITICAL: nvidia-smi not found. NVIDIA driver may not be installed.${NC}"
    exit 1
fi

# Check 2: Driver version
echo ""
echo "Check 2: NVIDIA driver version"
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
if [ -n "$DRIVER_VERSION" ]; then
    add_check "driver_version" "pass" "Driver version: $DRIVER_VERSION"
else
    add_check "driver_version" "fail" "Could not determine driver version"
fi

# Check 3: GPU count and detection
echo ""
echo "Check 3: GPU detection"
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
if [ "$GPU_COUNT" -gt 0 ] 2>/dev/null; then
    add_check "gpu_detection" "pass" "$GPU_COUNT GPU(s) detected"
else
    add_check "gpu_detection" "fail" "No GPUs detected"
fi

# Check 4: GPU status
echo ""
echo "Check 4: GPU status and health"
GPU_INFO=$(nvidia-smi --query-gpu=index,name,temperature.gpu,power.draw,memory.used,memory.total --format=csv,noheader)
add_check "gpu_status" "pass" "GPU information collected"

# Check 5: CUDA version
echo ""
echo "Check 5: CUDA version"
if nvidia-smi | grep -q "CUDA Version"; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    add_check "cuda_version" "pass" "CUDA Version: $CUDA_VERSION"
else
    add_check "cuda_version" "warn" "Could not determine CUDA version"
fi

# Check 6: GPU temperature check
echo ""
echo "Check 6: GPU temperature"
TEMPS=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)
MAX_TEMP=$(echo "$TEMPS" | sort -nr | head -1)
if [ "$MAX_TEMP" -lt 85 ] 2>/dev/null; then
    add_check "temperature_check" "pass" "Max temperature: ${MAX_TEMP}°C (OK)"
elif [ "$MAX_TEMP" -lt 95 ] 2>/dev/null; then
    add_check "temperature_check" "warn" "Max temperature: ${MAX_TEMP}°C (Warning: High)"
    OVERALL_STATUS="warn"
else
    add_check "temperature_check" "fail" "Max temperature: ${MAX_TEMP}°C (Critical)"
fi

# Check 7: PCIe link status
echo ""
echo "Check 7: PCIe link status"
PCIE_INFO=$(nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv,noheader)
add_check "pcie_link" "pass" "PCIe link information collected"

# Check 8: ECC errors (if supported)
echo ""
echo "Check 8: ECC memory errors"
if nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total --format=csv,noheader 2>/dev/null | grep -q "N/A"; then
    add_check "ecc_errors" "pass" "ECC not supported or no errors"
else
    ECC_ERRORS=$(nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits | paste -sd+ | bc)
    if [ "$ECC_ERRORS" -eq 0 ] 2>/dev/null; then
        add_check "ecc_errors" "pass" "No ECC errors detected"
    else
        add_check "ecc_errors" "fail" "$ECC_ERRORS uncorrected ECC errors detected"
    fi
fi

# Generate JSON output
cat > "$OUTPUT_FILE" << EOF
{
  "validation_type": "quick_check",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "hostname": "$(hostname)",
  "overall_status": "$OVERALL_STATUS",
  "driver_version": "$DRIVER_VERSION",
  "gpu_count": $GPU_COUNT,
  "checks": [
    $(IFS=,; echo "${CHECKS[*]}")
  ],
  "gpu_details": $(nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free,memory.used,temperature.gpu,utilization.gpu,power.draw,pcie.link.gen.current,pcie.link.width.current --format=csv,nounits | python3 -c "
import sys, csv, json
reader = csv.DictReader(sys.stdin)
print(json.dumps(list(reader), indent=2))
")
}
EOF

# Display summary
echo ""
echo "========================================"
echo "GPU Quick Validation Summary"
echo "========================================"
echo "Status: $OVERALL_STATUS"
echo "GPUs Detected: $GPU_COUNT"
echo "Driver Version: $DRIVER_VERSION"
echo "Report saved to: $OUTPUT_FILE"
echo ""

if [ "$OVERALL_STATUS" == "pass" ]; then
    echo -e "${GREEN}All checks passed!${NC}"
    exit 0
elif [ "$OVERALL_STATUS" == "warn" ]; then
    echo -e "${YELLOW}Validation completed with warnings.${NC}"
    exit 0
else
    echo -e "${RED}Validation failed! Please check the errors above.${NC}"
    exit 1
fi

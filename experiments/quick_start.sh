#!/bin/bash
# Quick start script for adversarial attack experiments

set -e

# Default values
IMAGE_ID=2332
CONFIG="configs/attack_config.yaml"
OUTPUT_DIR="experiments/results/dual_target_attack"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --image-id)
            IMAGE_ID="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: ./quick_start.sh [--image-id ID] [--config PATH] [--output-dir DIR]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================="
echo "HieRFE Dual-Target Adversarial Attack"
echo "========================================="
echo "Image ID: $IMAGE_ID"
echo "Config: $CONFIG"
echo "Output: $OUTPUT_DIR"
echo "========================================="
echo ""

# Run attack
python experiments/run_attack.py \
    --config "$CONFIG" \
    --image-id "$IMAGE_ID" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "========================================="
echo "Attack complete! Results saved to:"
echo "  $OUTPUT_DIR"
echo "========================================="


#!/bin/bash

SEEDS=(0 21 42 2541 3407)
# SEEDS=(0 )
ALLOWED_MODEL=("s" "m" "l" "x")
# ALLOWED_MODEL=("s")

for variant in "${ALLOWED_MODEL[@]}"; do
    for seed in "${SEEDS[@]}"; do
        python tools/deployment/export_onnx.py \
        -c configs/rtv4/rtv4_hgnetv2_${variant}_coco.yml \
        -r outputs/rtv4_hgnetv2_${variant}_coco/${seed}/last.pth \
        --output-dir onnx_models \
        --check \
        --simplify \
        # --dynamic
    done
done

#!/bin/bash

ALLOWED_MODEL=("s" "m" "l" "x")

# Function to check if a single argument is in the allowed list
is_in_list() {
  local value=$1
  shift
  local list=("$@")
  for item in "${list[@]}"; do
    if [[ "$item" == "$value" ]]; then
      return 0 # Found
    fi
  done
  return 1 # Not found
}

# Function to display usage
usage() {
  echo "Usage: $0 -m MODEL_VARIENT [-p PTH_NAME]"
  echo "  -m MODEL_VARIENT  Model variant (s, m, l, x)"
  echo "  -p PTH_NAME       Checkpoint name (last, best_stg1, best_stg2), default: last"
  echo ""
  echo "Example: $0 -m s -p last"
  exit 1
}

# Parse command line arguments
while getopts "m:p:" opt; do
  case $opt in
    m)
      MODEL_VARIENT=$OPTARG
      ;;
    p)
      PTH_NAME=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      usage
      ;;
  esac
done

# Check if MODEL_VARIENT is provided
if [ -z "$MODEL_VARIENT" ]; then
  echo "Error: MODEL_VARIENT is required"
  usage
fi

# Validate MODEL_VARIENT
if ! is_in_list "$MODEL_VARIENT" "${ALLOWED_MODEL[@]}"; then
  echo "Error: Invalid MODEL_VARIENT '$MODEL_VARIENT'. Allowed values: ${ALLOWED_MODEL[*]}"
  exit 1
fi

# Set default PTH_NAME if not provided
PTH_NAME=${PTH_NAME:-last}

export CUDA_VISIBLE_DEVICES="0,"

mkdir -p outputs/test/rtv4_hgnetv2_${MODEL_VARIENT}_coco
for seed in 0 21 42 2541 3407; do
    python train.py \
      --use-amp \
      --test-only \
      --output-dir outputs/test/rtv4_hgnetv2_${MODEL_VARIENT}_coco \
      --seed ${seed} \
      -c configs/rtv4/rtv4_hgnetv2_${MODEL_VARIENT}_coco.yml \
      -r outputs/rtv4_hgnetv2_${MODEL_VARIENT}_coco/${seed}/${PTH_NAME}.pth \
      > outputs/rtv4_hgnetv2_${MODEL_VARIENT}_coco/${seed}/test.log
done

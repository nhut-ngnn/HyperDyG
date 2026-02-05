#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="feature"
DATASET="IEMOCAP"
NUM_CLASSES=4

SAVE_DIR="logs"
MODALITY="both" 
CROSS_ATTN_BLOCKS=1

NOISE_DIR=""
NOISE_PATTERN="_test_"

MODEL_GLOB="saved_model/${DATASET}_${NUM_CLASSES}class_*.pt"

mapfile -t MODEL_PATHS < <(ls -1 $MODEL_GLOB 2>/dev/null || true)
if ((${#MODEL_PATHS[@]} == 0)); then
  echo "No model checkpoints found for glob: $MODEL_GLOB" >&2
  echo "Set MODEL_GLOB or drop a .pt file into saved_model/." >&2
  exit 1
fi

for MODEL_PATH in "${MODEL_PATHS[@]}"; do
  printf '\n=== Running prediction for %s ===\n' "$MODEL_PATH"
  cmd=(
    python3 trainer/predict.py
    --data_dir "$DATA_DIR"
    --model_path "$MODEL_PATH"
    --dataset "$DATASET"
    --num_classes "$NUM_CLASSES"
    --save_dir "$SAVE_DIR"
    --modality "$MODALITY"
    --cross_attn_blocks "$CROSS_ATTN_BLOCKS"
  )

  if [[ -n "$NOISE_DIR" ]]; then
    cmd+=(--noise_dir "$NOISE_DIR" --noise_pattern "$NOISE_PATTERN")
  fi

  if (($#)); then
    cmd+=("$@")
  fi

  "${cmd[@]}"
done

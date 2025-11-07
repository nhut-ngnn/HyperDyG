#!/usr/bin/env bash

set -euo pipefail

DATA_DIR="/home/minhnhutngnn/GloMER/features_output/"
DATASET="IEMOCAP"
NUM_CLASSES=4
MODEL_TYPE="HyperDyG"

EPOCHS=50
BATCH_SIZE=128

SUP_EPOCHS=50
SEMI_EPOCHS=50

PSEUDO_THRESHOLD=0.95
UNLABELED_BATCH_SIZE=32

RATIOS=(0)

for RATIO in "${RATIOS[@]}"; do
  printf '\n=== Running training with UNLABELED_RATIO=%s ===\n' "$RATIO"
  SANITIZED_RATIO=${RATIO//./p}
  OUTPUT_SUFFIX="${DATASET}_${NUM_CLASSES}class_ratio${SANITIZED_RATIO}"
  cmd=(
    python trainer/train_semi.py
    --data_dir "$DATA_DIR"
    --dataset "$DATASET"
    --model_type "$MODEL_TYPE"
    --num_classes "$NUM_CLASSES"
    --epochs "$EPOCHS"
    --consistency_weight 0.7
    --batch_size "$BATCH_SIZE"
    --supervised_epochs "$SUP_EPOCHS"
    --semi_epochs "$SEMI_EPOCHS"
    --pseudo_threshold "$PSEUDO_THRESHOLD"
    --unlabeled_batch_size "$UNLABELED_BATCH_SIZE"
    --unlabeled_ratio "$RATIO"
    --results_suffix "$OUTPUT_SUFFIX"
  )
  if (($#)); then
    cmd+=("$@")
  fi
  "${cmd[@]}"
done

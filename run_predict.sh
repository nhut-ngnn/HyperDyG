#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="feature"
DATASET="IEMOCAP"
NUM_CLASSES=4

BATCH_SIZE=128
SUP_EPOCHS=30
SEMI_EPOCHS=50

PSEUDO_THRESHOLD=0.95
UNLABELED_RATIO=0.7
CONSISTENCY_WEIGHT=0.1

THRESHOLDS=(0.88 0.90 0.92 0.94 0.96)

for THRESHOLD in "${THRESHOLDS[@]}"; do
  printf '\n=== Running training with PSEUDO WEIGHT THRESHOLD=%s ===\n' "$THRESHOLD"
  SANITIZED_THRESHOLD=${THRESHOLD/./p}
  OUTPUT_SUFFIX="0p1/${DATASET}_${NUM_CLASSES}class_thresh${SANITIZED_THRESHOLD}"

  cmd=(
    python3 trainer/train.py
    --data_dir "$DATA_DIR"
    --dataset "$DATASET"
    --num_classes "$NUM_CLASSES"
    --batch_size "$BATCH_SIZE"
    --supervised_epochs "$SUP_EPOCHS"
    --semi_epochs "$SEMI_EPOCHS"
    --pseudo_threshold "$THRESHOLDS"
    --unlabeled_ratio "$UNLABELED_RATIO"
    --pseudo_weight 0.1
    --results_suffix "$OUTPUT_SUFFIX"
  )

  if (($#)); then
    cmd+=("$@")
  fi

  "${cmd[@]}"
done

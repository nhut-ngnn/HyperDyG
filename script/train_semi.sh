#!/usr/bin/env bash
# Run semi-supervised experiments for a set of unlabeled ratios
# Adjust DATA_DIR to your path if needed
DATA_DIR="features_output/"
MODEL_TYPE="HyperDyG"
DATASET="IEMOCAP"
NUM_CLASSES=4
SEEDS="42,52,103,128,923"
UNLABELED_BATCH=128
SUP_EPOCHS=50
SEMI_EPOCHS=50
CONS_WEIGHT=0.7
PSEUDO_THRESH=0.95

RATIOS=(0)

for r in "${RATIOS[@]}"; do
  echo "==== Running unlabeled_ratio=${r} ===="
  python trainer/train_semi.py \
    --data_dir "${DATA_DIR}" \
    --dataset ${DATASET} \
    --num_classes ${NUM_CLASSES} \
    --unlabeled_ratio ${r} \
    --consistency_weight ${CONS_WEIGHT} \
    --pseudo_threshold ${PSEUDO_THRESH} \
    --supervised_epochs ${SUP_EPOCHS} \
    --semi_epochs ${SEMI_EPOCHS} \
    --unlabeled_batch_size ${UNLABELED_BATCH} \
    --model_type ${MODEL_TYPE} \
    --results_suffix "IEMOCAP_${NUM_CLASSES}class_unlabeled${r}" \
    | tee "logs/run_semi_ratio_${r}.log"
done

echo "All ratios completed. Logs are in logs/"

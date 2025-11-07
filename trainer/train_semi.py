import os
import sys
import torch
import pickle
import argparse
import random
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.utils import set_seed, train_and_evaluate
from src.GloMER.Model import CrossModalContrastiveModel
from src.MemoCMT.Model import MemoCMT, Config as MemoConfig
from src.FleSER.Model import FlexibleMMSER
from src.HyperDyG.Model import HyperDyG
from torch.utils.data import TensorDataset, DataLoader


def _ensure_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.float()
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    return torch.tensor(x, dtype=torch.float32)


def build_tensor_dataset(data, label_map=None):
    if label_map is None:
        label_map = {}

    next_index = len(label_map)
    text_tensors, audio_tensors = [], []
    label_tensors, confidence_tensors, pseudo_flags = [], [], []

    for item in data:
        base_text = _ensure_tensor(item['text_embed'])
        base_audio = _ensure_tensor(item['audio_embed'])

        text_tensors.append(base_text)
        audio_tensors.append(base_audio)

        label_value = item.get('label', -1)
        if isinstance(label_value, str):
            normalized_label = label_value.strip().lower()
            if normalized_label not in label_map:
                label_map[normalized_label] = next_index
                next_index += 1
            label_idx = label_map[normalized_label]
        else:
            label_idx = int(label_value)

        label_tensors.append(torch.tensor(label_idx, dtype=torch.long))

        confidence_value = item.get('confidence', 1.0)
        if confidence_value is None:
            confidence_value = 1.0
        confidence_tensors.append(torch.tensor(float(confidence_value), dtype=torch.float32))
        pseudo_flags.append(torch.tensor(1.0 if item.get('is_pseudo', False) else 0.0, dtype=torch.float32))

    text_tensor = torch.stack(text_tensors)
    audio_tensor = torch.stack(audio_tensors)
    label_tensor = torch.stack(label_tensors)
    confidence_tensor = torch.stack(confidence_tensors)
    pseudo_tensor = torch.stack(pseudo_flags)

    dataset = TensorDataset(
        text_tensor,
        audio_tensor,
        label_tensor,
        confidence_tensor,
        pseudo_tensor
    )

    return dataset, label_map


def build_unlabeled_dataset(data):
    if not data:
        return None

    text_tensors = [_ensure_tensor(item['text_embed']) for item in data]
    audio_tensors = [_ensure_tensor(item['audio_embed']) for item in data]

    return TensorDataset(torch.stack(text_tensors), torch.stack(audio_tensors))


def clone_samples(samples, is_pseudo=None, confidence=None):
    cloned = []
    for item in samples:
        new_item = dict(item)
        if is_pseudo is not None:
            new_item["is_pseudo"] = is_pseudo
        if confidence is not None or new_item.get("confidence") is None:
            new_item["confidence"] = confidence if confidence is not None else 1.0
        cloned.append(new_item)
    return cloned


def split_labeled_unlabeled(samples, ratio, seed=None):
    if not samples:
        return [], []

    ratio = max(0.0, min(1.0, float(ratio)))
    if ratio <= 0.0:
        return clone_samples(samples, is_pseudo=False, confidence=1.0), []
    if ratio >= 1.0:
        return [], clone_samples(samples, is_pseudo=True, confidence=0.0)

    indices = list(range(len(samples)))
    rng = random.Random(seed) if seed is not None else random.Random()
    rng.shuffle(indices)

    split_idx = int((1.0 - ratio) * len(samples))
    labeled_indices = indices[:split_idx]
    unlabeled_indices = indices[split_idx:]

    labeled = [
        dict(samples[idx], is_pseudo=False, confidence=1.0)
        for idx in labeled_indices
    ]
    unlabeled = [
        dict(samples[idx], is_pseudo=True, confidence=0.0)
        for idx in unlabeled_indices
    ]

    return labeled, unlabeled

def combined_loss(outputs, labels, ce_loss, confidences=None):
    logits = outputs["logits"]

    losses = ce_loss(logits, labels)   # [N]  

    if confidences is not None:
        weights = confidences.float()
        return (losses * weights).sum() / (weights.sum() + 1e-8)
    else:
        return losses.mean()


def get_filenames(data_dir, dataset, num_classes):
    if dataset == "IEMOCAP":
        assert num_classes == 4, "IEMOCAP now only supports 4 classes."
        prefix = "IEMOCAP_BERT_WavLM"
    elif dataset == "ViSEC":
        assert num_classes == 4, "ViSEC now only supports 4 classes."
        prefix = "ViSEC_PhoBERT_WavLM"
    elif dataset == "ESD":
        assert num_classes == 5, "ESD uses 5 classes."
        prefix = "ESD_BERT_WavLM"
    else:
        raise ValueError("Dataset must be either 'IEMOCAP', 'ViSEC', or 'ESD'.")

    return {
        "train": os.path.join(data_dir, f"{prefix}_train.pkl"),
        "val": os.path.join(data_dir, f"{prefix}_val.pkl"),
        "test": os.path.join(data_dir, f"{prefix}_test.pkl"),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train GloMER on IEMOCAP, MELD or ESD with 5 seeds")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=["IEMOCAP", "ViSEC", "ESD"])
    parser.add_argument("--num_classes", type=int, required=True, choices=[4, 5, 7],
                        help="4 for IEMOCAP, 5 for ESD, 7 for MELD")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training and evaluation")
    parser.add_argument("--labeled_train_path", type=str, default=None,
                        help="Optional path to the labeled training PKL file. Defaults to the dataset train file.")
    parser.add_argument("--pseudo_train_path", type=str, default=None,
                        help="Path to the unlabeled/pseudo data PKL file to enable semi-supervised training.")
    parser.add_argument("--supervised_epochs", type=int, default=None,
                        help="Number of epochs for the supervised pretraining stage. Defaults to --epochs.")
    parser.add_argument("--semi_epochs", type=int, default=None,
                        help="Number of epochs for the semi-supervised fine-tuning stage. Defaults to --epochs.")
    parser.add_argument("--pseudo_threshold", type=float, default=0.9,
                        help="Confidence threshold to accept pseudo-labeled samples.")
    parser.add_argument("--max_pseudo_ratio", type=float, default=None,
                        help="Optional cap on pseudo samples per class as a multiple of labeled samples (e.g., 0.5).")
    parser.add_argument("--pseudo_weight", type=float, default=1.0,
                        help="Additional scaling applied to pseudo sample confidences when computing the loss.")
    parser.add_argument("--unlabeled_ratio", type=float, default=0.0,
                        help="Fraction of training data to treat as unlabeled for semi-supervised learning. Ignored if --pseudo_train_path is provided.")
    parser.add_argument("--split_seed", type=int, default=None,
                        help="Seed used when splitting labeled/unlabeled data. Defaults to the current training seed.")
    parser.add_argument("--results_suffix", type=str, default=None,
                        help="Optional custom suffix for the aggregated results CSV filename.")
    parser.add_argument("--consistency_weight", type=float, default=1.0,
                        help="Weight applied to the consistency loss on unlabeled data.")
    parser.add_argument("--weak_word_dropout", type=float, default=0.1,
                        help="Word dropout probability for weak augmentation.")
    parser.add_argument("--strong_word_dropout", type=float, default=0.3,
                        help="Word dropout probability for strong augmentation.")
    parser.add_argument("--weak_audio_noise_std", type=float, default=0.01,
                        help="Gaussian noise std for audio during weak augmentation.")
    parser.add_argument("--strong_audio_noise_std", type=float, default=0.05,
                        help="Gaussian noise std for audio during strong augmentation.")
    parser.add_argument("--weak_text_noise_std", type=float, default=0.0,
                        help="Gaussian noise std for text during weak augmentation.")
    parser.add_argument("--strong_text_noise_std", type=float, default=0.0,
                        help="Gaussian noise std for text during strong augmentation.")
    parser.add_argument("--unlabeled_batch_size", type=int, default=None,
                        help="Batch size for the unlabeled data loader. Defaults to --batch_size.")
    parser.add_argument(
        "--disable_augmentation",
        action="store_true",
        help="Skip generating weak/strong text and audio augmentations during training."
    )
    parser.add_argument(
        "--supervised_strong_aug",
        action="store_true",
        help="Apply strong augmentation to labeled batches when no unlabeled data is used."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["GloMER", "MemoCMT", "FleSER", "HyperDyG"],
        default="GloMER",
        help="Choose which model architecture to train: GloMER (default), MemoCMT or FleSER"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = [42, 52, 103, 128, 923]
    # seeds = [42]

    supervised_epochs = args.supervised_epochs or args.epochs
    semi_epochs = args.semi_epochs or args.epochs
    auto_split_ratio = max(0.0, float(args.unlabeled_ratio or 0.0))
    use_manual_pseudo = args.pseudo_train_path is not None
    auto_split_enabled = (not use_manual_pseudo) and auto_split_ratio > 0.0
    use_pseudo = use_manual_pseudo or auto_split_enabled
    augmentation_enabled = not args.disable_augmentation
    supervised_aug_active = False
    if args.supervised_strong_aug:
        if use_pseudo:
            print("[WARN] --supervised_strong_aug is ignored because pseudo/unlabeled data is enabled.")
        elif not augmentation_enabled:
            print("[WARN] --disable_augmentation overrides --supervised_strong_aug; no supervised augmentation will run.")
        else:
            supervised_aug_active = True
    ratio_name_component = f"unlabeled{auto_split_ratio:.4f}".replace(".", "p")

    if use_manual_pseudo and auto_split_ratio > 0.0:
        print("[INFO] --unlabeled_ratio is ignored because --pseudo_train_path was provided.")

    filenames = get_filenames(args.data_dir, args.dataset, args.num_classes)
    train_path = args.labeled_train_path or filenames["train"]
    with open(train_path, "rb") as f:
        train_raw_full = pickle.load(f)
    with open(filenames["val"], "rb") as f:
        val_raw_full = pickle.load(f)
    with open(filenames["test"], "rb") as f:
        test_raw_full = pickle.load(f)
    manual_pseudo_raw = []
    if use_manual_pseudo:
        with open(args.pseudo_train_path, "rb") as f:
            manual_pseudo_raw = pickle.load(f)

    wa_list, ua_list, wf1_list, uf1_list = [], [], [], []
    # track supervised-stage metrics and semi-supervised-stage metrics separately
    sup_wa_list, sup_ua_list, sup_wf1_list, sup_uf1_list = [], [], [], []
    semi_wa_list, semi_ua_list, semi_wf1_list, semi_uf1_list = [], [], [], []

    for seed in seeds:
        set_seed(seed)
        print(f"\n=== Running with seed {seed} ===")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        split_seed_value = args.split_seed if (auto_split_enabled and args.split_seed is not None) else None

        if auto_split_enabled:
            if split_seed_value is None:
                split_seed_value = seed
            labeled_samples, pseudo_candidates = split_labeled_unlabeled(
                train_raw_full, auto_split_ratio, split_seed_value
            )
        else:
            labeled_samples = clone_samples(train_raw_full, is_pseudo=False, confidence=1.0)
            pseudo_candidates = clone_samples(manual_pseudo_raw, is_pseudo=True) if use_manual_pseudo else []

        label_map = {}
        train_dataset, label_map = build_tensor_dataset(labeled_samples, label_map=label_map)
        val_dataset, label_map = build_tensor_dataset(
            clone_samples(val_raw_full, is_pseudo=False, confidence=1.0),
            label_map=label_map
        )
        test_dataset, label_map = build_tensor_dataset(
            clone_samples(test_raw_full, is_pseudo=False, confidence=1.0),
            label_map=label_map
        )

        pseudo_raw = list(pseudo_candidates) if pseudo_candidates else []
        pseudo_available = len(pseudo_raw)

        # infer input dims from built datasets so model matches feature extractor output
        inferred_text_dim = train_dataset.tensors[0].size(1)
        inferred_audio_dim = train_dataset.tensors[1].size(1)

        if args.model_type == "GloMER":
            model = CrossModalContrastiveModel(
                text_input_dim=inferred_text_dim,
                audio_input_dim=inferred_audio_dim,
                fusion_dim=512,
                projection_dim=256,
                num_heads=8,
                dropout=0.2,
                linear_layer_dims=[512, 128],
                num_classes=args.num_classes
            ).to(device)
        elif args.model_type == "MemoCMT":
            # Build a lightweight MemoCMT config using inferred dims and reasonable defaults
            memo_cfg = MemoConfig(
                text_encoder_dim=inferred_text_dim,
                audio_encoder_dim=inferred_audio_dim,
                fusion_dim=512,
                num_attention_head=8,
                dropout=0.2,
                linear_layer_output=[512, 128],
                num_classes=args.num_classes,
                fusion_head_output_type="mean",
            )
            model = MemoCMT(memo_cfg, device=device).to(device)
        elif args.model_type == "FleSER":
            # Instantiate FleSER (FlexibleMMSER) using inferred dims
            model = FlexibleMMSER(
                text_input_dim=inferred_text_dim,
                audio_input_dim=inferred_audio_dim,
                num_classes=args.num_classes,
                fusion_method='self_attention',
                alpha=0.5,
                dropout_rate=0.2,
                use_layernorm=True,
                text_only=False,
                hidden_dim=512,
                proj_dim=256,
                num_heads=8
            ).to(device)
        elif args.model_type == "HyperDyG":
            # Instantiate HyperDyG using inferred dims and reasonable defaults
            model = HyperDyG(
                text_input_dim=inferred_text_dim,
                audio_input_dim=inferred_audio_dim,
                fusion_dim=512,
                projection_dim=256,
                num_heads=8,
                dropout=0.2,
                linear_layer_dims=[512, 128],
                num_classes=args.num_classes,
                hypergraph_k_text=5,
                hypergraph_k_audio=5,
                hypergraph_threshold=0.5,
                use_rgcn=True,
                use_gtr=True,
                num_relations=3
            ).to(device)
        else:
            # Fallback (should not happen because argparse restricts choices)
            raise ValueError(f"Unsupported model_type: {args.model_type}")

        labeled_count = train_dataset.tensors[0].size(0)
        val_count = val_dataset.tensors[0].size(0)
        test_count = test_dataset.tensors[0].size(0)

        train_labels = train_dataset.tensors[2]
        class_sample_count = torch.bincount(train_labels, minlength=args.num_classes).float()
        class_sample_count[class_sample_count == 0] = 1.0
        class_weights = 1.0 / class_sample_count
        class_weights = class_weights / class_weights.sum() * args.num_classes
        class_weights = class_weights.to(device)

        ce_loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='none')
        loss_fn = lambda out, y, conf: combined_loss(out, y, ce_loss_fn, confidences=conf)

        optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=1e-2)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)

        save_path_supervised = (
            f"ablation_models/{args.dataset}_{args.num_classes}class_{ratio_name_component}_{args.model_type}_seed{seed}.pt"
        )
        unlabeled_batch_size_value = args.unlabeled_batch_size or args.batch_size

        tags_base = [args.dataset, f"{args.num_classes}class"]
        common_context = {
            "dataset": args.dataset,
            "num_classes": args.num_classes,
            "batch_size": args.batch_size,
            "group": f"{args.dataset}_{args.num_classes}class",
            "project": f"{args.model_type}-EmotionRecognition-{args.dataset}-Ablation",
            "experiment_name": f"{args.dataset}_{args.num_classes}class",
            "seed": seed,
            "pseudo_enabled": use_pseudo,
            "use_augmentation": augmentation_enabled if use_pseudo else supervised_aug_active,
            "supervised_strong_aug": supervised_aug_active if not use_pseudo else None,
            "pseudo_threshold": args.pseudo_threshold if use_pseudo else None,
            "consistency_weight": args.consistency_weight if (use_pseudo or supervised_aug_active) else None,
            "weak_word_dropout": args.weak_word_dropout if use_pseudo else None,
            "strong_word_dropout": args.strong_word_dropout if (use_pseudo or supervised_aug_active) else None,
            "weak_audio_noise_std": args.weak_audio_noise_std if use_pseudo else None,
            "strong_audio_noise_std": args.strong_audio_noise_std if (use_pseudo or supervised_aug_active) else None,
            "weak_text_noise_std": args.weak_text_noise_std if use_pseudo else None,
            "strong_text_noise_std": args.strong_text_noise_std if (use_pseudo or supervised_aug_active) else None,
            "unlabeled_batch_size": unlabeled_batch_size_value if use_pseudo else None,
            "labeled_train_path": train_path,
            "pseudo_train_path": (
                args.pseudo_train_path if use_manual_pseudo else ("split_in_memory" if auto_split_enabled else None)
            ),
            "labeled_samples": labeled_count,
            "val_samples": val_count,
            "test_samples": test_count,
            "pseudo_available": pseudo_available,
            "unlabeled_ratio": auto_split_ratio if auto_split_enabled else None,
            "split_seed": split_seed_value,
            "results_suffix": args.results_suffix
        }

        supervised_metrics = train_and_evaluate(
            model, train_dataset, val_dataset, test_dataset,
            optimizer, scheduler,
            loss_fn, 
            epochs=supervised_epochs,
            save_path=save_path_supervised,
            dataset=args.dataset,
            seed=seed,alpha=0.3,
            batch_size=args.batch_size,
            log_context={
                **common_context,
                "stage": "supervised",
                "tags": tags_base + ["supervised"],
                "unlabeled_ratio": None,
                "use_augmentation": supervised_aug_active
            },
            supervised_aug_cfg=(
                {
                    "enabled": True,
                    "word_dropout": args.strong_word_dropout,
                    "audio_noise_std": args.strong_audio_noise_std,
                    "text_noise_std": args.strong_text_noise_std,
                    "consistency_weight": args.consistency_weight
                } if supervised_aug_active else None
            )
        )

        final_metrics = supervised_metrics

        # record supervised-stage metrics for this seed
        sup_wa_list.append(supervised_metrics.get("test_WA", float('nan')))
        sup_ua_list.append(supervised_metrics.get("test_UA", float('nan')))
        sup_wf1_list.append(supervised_metrics.get("test_WF1", float('nan')))
        sup_uf1_list.append(supervised_metrics.get("test_UF1", float('nan')))

        if use_pseudo and pseudo_available > 0 and augmentation_enabled:
            unlabeled_dataset = build_unlabeled_dataset(pseudo_raw)
            if unlabeled_dataset is None or len(unlabeled_dataset) == 0:
                print("No unlabeled samples available for semi-supervised stage; retaining supervised-only model for this seed.")
            else:
                optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=1e-2)
                scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)

                ce_loss_fn_semi = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='none')
                loss_fn_semi = lambda out, y, conf: combined_loss(
                    out, y, ce_loss_fn_semi, confidences=conf
                )

                save_path_semi = (
                    f"saved_model/{args.dataset}_{args.num_classes}class_{ratio_name_component}_{args.model_type}_seed{seed}_semi.pt"
                )
                unlabeled_cfg = {
                    "consistency_weight": args.consistency_weight,
                    "pseudo_threshold": args.pseudo_threshold,
                    "weak_word_dropout": args.weak_word_dropout,
                    "strong_word_dropout": args.strong_word_dropout,
                    "weak_audio_noise_std": args.weak_audio_noise_std,
                    "strong_audio_noise_std": args.strong_audio_noise_std,
                    "weak_text_noise_std": args.weak_text_noise_std,
                    "strong_text_noise_std": args.strong_text_noise_std,
                    "unlabeled_batch_size": unlabeled_batch_size_value
                }

                final_metrics = train_and_evaluate(
                    model, train_dataset, val_dataset, test_dataset,
                    optimizer, scheduler,
                    loss_fn_semi,alpha=0.3,
                    epochs=semi_epochs,
                    save_path=save_path_semi,
                    seed=seed,
                    batch_size=args.batch_size,
                    log_context={
                        **common_context,
                        "stage": "semi_supervised",
                        "tags": tags_base + ["semi", "pseudo"],
                        "use_augmentation": augmentation_enabled,
                        "combined_train_samples": labeled_count
                    },
                    unlabeled_dataset=unlabeled_dataset,
                    unlabeled_cfg=unlabeled_cfg
                )
                semi_wa_list.append(final_metrics.get("test_WA", float('nan')))
                semi_ua_list.append(final_metrics.get("test_UA", float('nan')))
                semi_wf1_list.append(final_metrics.get("test_WF1", float('nan')))
                semi_uf1_list.append(final_metrics.get("test_UF1", float('nan')))
        elif use_pseudo and pseudo_available > 0 and not augmentation_enabled:
            print("Augmentation disabled via --disable_augmentation; skipping semi-supervised consistency stage for this seed.")
        elif use_pseudo:
            print("No unlabeled samples available for pseudo labeling; skipping semi-supervised stage for this seed.")

        print(
            f"Seed {seed} - Final WA: {final_metrics['test_WA']:.4f}, UA: {final_metrics['test_UA']:.4f}, "
            f"WF1: {final_metrics['test_WF1']:.4f}, UF1: {final_metrics['test_UF1']:.4f}"
        )

    wa_list.append(final_metrics["test_WA"])
    ua_list.append(final_metrics["test_UA"])
    wf1_list.append(final_metrics["test_WF1"])
    uf1_list.append(final_metrics["test_UF1"])

    print("\n=== Supervised-stage Average Results over 5 seeds ===")
    print(f"Sup WA:  {np.nanmean(sup_wa_list):.4f}, {np.nanstd(sup_wa_list, ddof=1):.4f}")
    print(f"Sup UA:  {np.nanmean(sup_ua_list):.4f}, {np.nanstd(sup_ua_list, ddof=1):.4f}")
    print(f"Sup WF1: {np.nanmean(sup_wf1_list):.4f}, {np.nanstd(sup_wf1_list, ddof=1):.4f}")
    print(f"Sup UF1: {np.nanmean(sup_uf1_list):.4f}, {np.nanstd(sup_uf1_list, ddof=1):.4f}")

    if len(semi_wa_list) > 0:
        print("\n=== Semi-supervised-stage Average Results over seeds that ran semi ===")
        print(f"Semi WA:  {np.nanmean(semi_wa_list):.4f}, {np.nanstd(semi_wa_list, ddof=1):.4f}")
        print(f"Semi UA:  {np.nanmean(semi_ua_list):.4f}, {np.nanstd(semi_ua_list, ddof=1):.4f}")
        print(f"Semi WF1: {np.nanmean(semi_wf1_list):.4f}, {np.nanstd(semi_wf1_list, ddof=1):.4f}")
        print(f"Semi UF1: {np.nanmean(semi_uf1_list):.4f}, {np.nanstd(semi_uf1_list, ddof=1):.4f}")

    results_df = pd.DataFrame({
        "Metric": ["Final_WA", "Final_UA", "Final_WF1", "Final_UF1",
                   "Sup_WA", "Sup_UA", "Sup_WF1", "Sup_UF1"],
        "Mean": [
            np.mean(wa_list), np.mean(ua_list), np.mean(wf1_list), np.mean(uf1_list),
            np.nanmean(sup_wa_list), np.nanmean(sup_ua_list), np.nanmean(sup_wf1_list), np.nanmean(sup_uf1_list)
        ],
        "Std": [
            np.std(wa_list, ddof=1), np.std(ua_list, ddof=1), np.std(wf1_list, ddof=1), np.std(uf1_list, ddof=1),
            np.nanstd(sup_wa_list, ddof=1), np.nanstd(sup_ua_list, ddof=1), np.nanstd(sup_wf1_list, ddof=1), np.nanstd(sup_uf1_list, ddof=1)
        ]
    })

    os.makedirs("results", exist_ok=True)
    if args.results_suffix:
        result_file_name = args.results_suffix
    else:
        result_suffix = f"{args.model_type}_5seeds" if not use_pseudo else f"{args.model_type}_SemiPseudo_5seeds"
        result_file_name = f"{args.dataset}_{args.num_classes}class_{result_suffix}"
    results_path = os.path.join("results", f"{result_file_name}.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")

if __name__ == "__main__":
    main()

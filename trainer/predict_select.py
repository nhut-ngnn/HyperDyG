import os
import sys
import time
import torch
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import Data
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    accuracy_score,
    f1_score
)
from sklearn.manifold import TSNE
from scipy.special import softmax
from thop import profile
from sklearn.metrics import precision_recall_fscore_support

import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.utils import set_seed
from src.GloMER.Model import CrossModalContrastiveModel
from src.MemoCMT.Model import MemoCMT, Config as MemoConfig
from src.FleSER.Model import FlexibleMMSER
from src.HyperDyG.Model import HyperDyG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "state"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        return obj
    return obj


def _strip_module_prefix(state_dict):
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def _infer_arch_from_state_dict(state_dict):
    keys = list(state_dict.keys())
    if not keys:
        return None
    hyper_patterns = ("hypergraph", "rgcn", "graph_transformer", "hyper_expand", "DualHypergraphModule")
    memo_patterns = ("text_encoder.proj", "audio_encoder.proj", "text_attention.in_proj_weight", "classifer.weight")
    fleser_patterns = ("projection.", "projection.text_encoder", "projection.audio_encoder", "fc.")
    glo_patterns = ("encoders.", "gmu.", "shared_proj.", "classifier.model")
    for k in keys[:200]:
        for p in hyper_patterns:
            if p in k:
                return "HyperDyG"
        for p in glo_patterns:
            if p in k:
                return "GloMER"
        for p in memo_patterns:
            if p in k:
                return "MemoCMT"
        for p in fleser_patterns:
            if p in k:
                return "FleSER"
    if any("encoders." in k for k in keys):
        return "GloMER"
    return None


def _infer_hyperdyg_dims(state_dict):

    candidates = [
        "gmu.audio_proj.weight",
        "gmu.text_proj.weight",
        "gmu.gate_proj.weight",
    ]
    for k in candidates:
        if k in state_dict:
            tensor = state_dict[k]
            if hasattr(tensor, 'shape') and len(tensor.shape) == 2:
                in_dim = tensor.shape[1]
                if in_dim % 2 == 0:
                    return in_dim // 2
    return None


def load_data(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data[0], tuple):
        raise ValueError(f"Loaded data is raw. Please provide feature-extracted .pkl: {pkl_path}")
    text = torch.stack([torch.tensor(item['text_embed']) for item in data])
    audio = torch.stack([torch.tensor(item['audio_embed']) for item in data])
    labels = torch.tensor([item['label'] for item in data])
    return Data(text_x=text, audio_x=audio, y=labels)


def compute_metrics(y_true, y_pred):
    wa = balanced_accuracy_score(y_true, y_pred)
    ua = accuracy_score(y_true, y_pred)
    wf1 = f1_score(y_true, y_pred, average="weighted")
    uf1 = f1_score(y_true, y_pred, average="macro")
    return wa, ua, wf1, uf1


def plot_confusion_matrix(y_true, y_pred, label_names, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm_percent, annot=True, fmt=".2f", cmap="Blues", cbar=True,
        xticklabels=label_names, yticklabels=label_names,
        annot_kws={"size": 14, "weight": "bold"} 
    )
    plt.xlabel("Predicted Label", fontsize=14, fontweight="bold")
    plt.ylabel("True Label", fontsize=14, fontweight="bold")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{save_path}.pdf", dpi=500, format="pdf")
    plt.close()
    print(f"Saved confusion matrix to {save_path}.pdf")


def plot_tsne(features, labels, label_names, save_path):
    tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto")
    reduced = tsne.fit_transform(features)

    plt.figure(figsize=(7, 6))
    for i, label_name in enumerate(label_names):
        idx = labels == i
        plt.scatter(
            reduced[idx, 0], reduced[idx, 1],
            label=label_name, s=10, alpha=0.7
        )
    plt.legend(fontsize=10)
    plt.xticks(fontsize=12, fontweight="bold")
    plt.yticks(fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{save_path}.pdf", dpi=500, format="pdf")
    plt.close()
    print(f"Saved t-SNE plot to {save_path}.pdf")


def parse_args():
    parser = argparse.ArgumentParser(description="Predict model on IEMOCAP, and ESD with selectable architecture")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing test.pkl")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model .pt")
    parser.add_argument("--dataset", type=str, required=True, choices=["IEMOCAP", "ESD"], help="Dataset")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of emotion classes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="logs", help="Directory to save outputs")
    parser.add_argument("--modality", type=str, choices=["both", "text", "audio"], default="both", help="Modality to predict with")
    parser.add_argument("--model_type", type=str, choices=["GloMER", "MemoCMT", "FleSER", "HyperDyG"], default="GloMER", help="Model architecture to load")
    parser.add_argument("--skip_missing", action="store_true", help="Skip missing seed checkpoints instead of erroring")
    parser.add_argument("--seeds", type=str, default="42,52,103,128,923",
                        help="Comma-separated seeds to aggregate results over (default: 5 seeds)")
    return parser.parse_args()


def get_filenames(data_dir, dataset, num_classes, split="test"):
    if dataset == "IEMOCAP":
        assert num_classes == 4, "IEMOCAP supports only 4 classes."
        prefix = f"IEMOCAP_BERT_WavLM"
    elif dataset == "ESD":
        assert num_classes == 5, "ESD uses 5 classes."
        prefix = "ESD_BERT_WavLM"
    else:
        raise ValueError("Dataset must be either 'IEMOCAP', or 'ESD'.")
    return os.path.join(data_dir, f"{prefix}_{split}.pkl")


def main():
    args = parse_args()
    set_seed(args.seed)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    os.makedirs(args.save_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.model_path))[0]
    suffix = f"_{args.modality}only" if args.modality != "both" else ""

    if args.dataset == "IEMOCAP":
        label_names = ["Angry", "Happy", "Sad", "Neutral"]
    elif args.dataset == "ESD":
        label_names = ["Angry", "Happy", "Sad", "Neutral", "Surprise"]

    print("\nLoading test data...")
    test_pkl = get_filenames(args.data_dir, args.dataset, args.num_classes, split="test")
    test_data = load_data(test_pkl).to(device)

    inferred_text_dim = test_data.text_x.size(1)
    inferred_audio_dim = test_data.audio_x.size(1)


    def resolve_model_path(base_path, seed):
        if "{seed}" in base_path:
            return base_path.format(seed=seed)
        if f"{seed}" in base_path and os.path.exists(base_path):
            return base_path
        base, ext = os.path.splitext(base_path)
        candidate = f"{base}_seed{seed}_semi{ext}"
        if os.path.exists(candidate):
            return candidate
        # fallback: if exact base_path exists, return it (single checkpoint used for all seeds)
        if os.path.exists(base_path):
            return base_path
        # final fallback: return candidate even if missing (will error when loading)
        return candidate

    per_seed_results = []
    per_seed_class_f1 = []
    per_seed_class_acc = []

    for seed in seeds:
        print(f"\n--- Processing seed {seed} ---")
        model_path_for_seed = resolve_model_path(args.model_path, seed)

        # If the resolved path doesn't exist, honor --skip_missing
        if not os.path.exists(model_path_for_seed):
            if args.skip_missing:
                print(f"Warning: checkpoint for seed {seed} not found at {model_path_for_seed}; skipping.")
                continue
            else:
                raise FileNotFoundError(f"Checkpoint not found: {model_path_for_seed}")

        print(f"Loading model from: {model_path_for_seed}")
        # load checkpoint then infer architecture from keys to avoid mismatched model/state
        ckpt_obj = torch.load(model_path_for_seed, map_location=device)
        state_dict = _extract_state_dict(ckpt_obj)
        state_dict = _strip_module_prefix(state_dict)
        inferred = _infer_arch_from_state_dict(state_dict)
        arch_to_use = inferred if inferred is not None else args.model_type

        if arch_to_use == "GloMER":
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
        elif arch_to_use == "MemoCMT":
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
        elif arch_to_use == "FleSER":
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
        elif arch_to_use == "HyperDyG":
            # try to infer fusion_dim from checkpoint; fallback to 512
            inferred_fusion = _infer_hyperdyg_dims(state_dict)
            fusion_dim_to_use = inferred_fusion if inferred_fusion is not None else 512
            if inferred_fusion is not None:
                print(f"Inferred HyperDyG fusion_dim={fusion_dim_to_use} from checkpoint")
            model = HyperDyG(
                text_input_dim=inferred_text_dim,
                audio_input_dim=inferred_audio_dim,
                fusion_dim=fusion_dim_to_use,
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
            # fallback: respect user-provided arg
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
                raise ValueError(f"Unsupported model_type: {args.model_type}")

        res = model.load_state_dict(state_dict, strict=False)
        missing = getattr(res, 'missing_keys', getattr(res, 'missing', []))
        unexpected = getattr(res, 'unexpected_keys', getattr(res, 'unexpected', []))
        if missing:
            print(f"Missing keys when loading checkpoint for seed {seed}: {missing}")
        if unexpected:
            print(f"Unexpected keys when loading checkpoint for seed {seed}: {unexpected}")
        model.eval()

        with torch.no_grad():
            if args.modality == "text":
                audio_dummy = torch.zeros_like(test_data.audio_x)
                outputs = model(test_data.text_x, audio_dummy, return_all=True)
            elif args.modality == "audio":
                text_dummy = torch.zeros_like(test_data.text_x)
                outputs = model(text_dummy, test_data.audio_x, return_all=True)
            else:
                outputs = model(test_data.text_x, test_data.audio_x, return_all=True)

            logits = outputs["logits"].cpu().numpy()
            probs = softmax(logits, axis=1)
            preds = np.argmax(probs, axis=1)
            labels = test_data.y.cpu().numpy()

        wa, ua, wf1, mf1 = compute_metrics(labels, preds)
        precision, recall, f1_per_class, support = precision_recall_fscore_support(labels, preds, labels=list(range(args.num_classes)), zero_division=0)

        per_seed_results.append({"seed": seed, "WA": wa, "UA": ua, "WF1": wf1, "MF1": mf1})
        per_seed_class_f1.append(f1_per_class)
        per_seed_class_acc.append(recall)

        report = classification_report(labels, preds, target_names=label_names, digits=4)
        seed_report_path = os.path.join(args.save_dir, f"{base_name}_seed{seed}_classification_report{suffix}.txt")
        with open(seed_report_path, "w") as f:
            f.write(report)
        print(f"Saved per-seed classification report to {seed_report_path}")

    per_seed_class_f1 = np.vstack(per_seed_class_f1)  # shape (n_seeds, n_classes)
    per_seed_class_acc = np.vstack(per_seed_class_acc)

    class_f1_mean = per_seed_class_f1.mean(axis=0)
    class_f1_std = per_seed_class_f1.std(axis=0, ddof=1) if per_seed_class_f1.shape[0] > 1 else np.zeros_like(class_f1_mean)
    class_acc_mean = per_seed_class_acc.mean(axis=0)
    class_acc_std = per_seed_class_acc.std(axis=0, ddof=1) if per_seed_class_acc.shape[0] > 1 else np.zeros_like(class_acc_mean)

    WA_list = [r["WA"] for r in per_seed_results]
    UA_list = [r["UA"] for r in per_seed_results]
    WF1_list = [r["WF1"] for r in per_seed_results]
    MF1_list = [r["MF1"] for r in per_seed_results]

    overall_summary = {
        "WA_mean": float(np.mean(WA_list)),
        "WA_std": float(np.std(WA_list, ddof=1)) if len(WA_list) > 1 else 0.0,
        "UA_mean": float(np.mean(UA_list)),
        "UA_std": float(np.std(UA_list, ddof=1)) if len(UA_list) > 1 else 0.0,
        "WF1_mean": float(np.mean(WF1_list)),
        "WF1_std": float(np.std(WF1_list, ddof=1)) if len(WF1_list) > 1 else 0.0,
        "MF1_mean": float(np.mean(MF1_list)),
        "MF1_std": float(np.std(MF1_list, ddof=1)) if len(MF1_list) > 1 else 0.0,
    }

    agg_path = os.path.join(args.save_dir, f"{base_name}_aggregated_over_seeds{suffix}.txt")
    with open(agg_path, "w") as f:
        f.write("Per-class Accuracy (recall) mean  std:\n")
        for i, name in enumerate(label_names):
            f.write(f"{name}: {class_acc_mean[i]:.4f}  {class_acc_std[i]:.4f}\n")
        f.write("\nPer-class F1 mean  std:\n")
        for i, name in enumerate(label_names):
            f.write(f"{name}: {class_f1_mean[i]:.4f}  {class_f1_std[i]:.4f}\n")
        f.write("\nOverall metrics mean  std over seeds:\n")
        for k, v in overall_summary.items():
            f.write(f"{k}: {v:.4f}\n")

    print(f"Saved aggregated report to {agg_path}")

    print("\n=== Aggregated results over seeds ===")
    print("Per-class Accuracy (recall) mean:")
    for i, name in enumerate(label_names):
        print(f"{name}: {class_acc_mean[i]:.4f}  {class_acc_std[i]:.4f}")
    print("\nPer-class F1 mean:")
    for i, name in enumerate(label_names):
        print(f"{name}: {class_f1_mean[i]:.4f}  {class_f1_std[i]:.4f}")
    print("\nOverall metrics:")
    for k, v in overall_summary.items():
        print(f"{k}: {v:.4f}")

    return


if __name__ == "__main__":
    main()

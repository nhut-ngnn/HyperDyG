import torch
from transformers import AutoTokenizer, AutoFeatureExtractor, BertTokenizer
from .model_encode import PhoBERTEmbeddingModel, WavLMEmbeddingModel, BERTEmbeddingModel
import os
PKL_DIR = "metadata"
OUTPUT_DIR = "feature"

# Fine-tuned checkpoints for IEMOCAP/ESD can be provided via env vars when available.
DEFAULT_TEXT_CKPT = None
DEFAULT_AUDIO_CKPT = None

TEXT_CKPT_PATH = os.getenv("TEXT_CKPT_PATH", DEFAULT_TEXT_CKPT)
AUDIO_CKPT_PATH = os.getenv("AUDIO_CKPT_PATH", DEFAULT_AUDIO_CKPT)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#TOKENIZER = AutoTokenizer.from_pretrained("vinai/phobert-base")
TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
AUDIO_PROCESSOR = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base")

TEXT_MODEL = PhoBERTEmbeddingModel(embedding_dim=768, projection_dim=512).to(device)
AUDIO_MODEL = WavLMEmbeddingModel(embedding_dim=768, projection_dim=512).to(device)


def _load_checkpoint(model, path, description):
    if not path:
        print(f"[WARN] No checkpoint path provided for {description}; using pretrained weights.")
        return
    if not os.path.isfile(path):
        print(f"[WARN] Checkpoint for {description} not found at '{path}'; using pretrained weights.")
        return
    try:
        state = torch.load(path, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)
        print(f"[INFO] Loaded {description} checkpoint from {path}")
    except Exception as exc:
        print(f"[WARN] Failed to load {description} checkpoint from '{path}': {exc}. Using pretrained weights.")


_load_checkpoint(TEXT_MODEL, TEXT_CKPT_PATH, "text encoder")
_load_checkpoint(AUDIO_MODEL, AUDIO_CKPT_PATH, "audio encoder")

TEXT_MODEL.eval()
AUDIO_MODEL.eval()

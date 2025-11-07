import torch
from transformers import AutoTokenizer, AutoFeatureExtractor, BertTokenizer
from .model_encode import PhoBERTEmbeddingModel, WavLMEmbeddingModel, BERTEmbeddingModel
import os
PKL_DIR = "metadata"
OUTPUT_DIR = "feature"

#DEFAULT_TEXT_CKPT = "/home/tri.pm/polyp/fptu/MinhNhut/fine_tuning/ViSEC/models/best_phobert_embeddings.pt"

DEFAULT_TEXT_CKPT = "/home/tri.pm/polyp/fptu/MinhNhut/fine_tuning/ESD/models/best_bert_embeddings.pt"
DEFAULT_AUDIO_CKPT = "/home/tri.pm/polyp/fptu/MinhNhut/fine_tuning/ESD/models/best_wavlm_embeddings.pt"

TEXT_CKPT_PATH = os.getenv("TEXT_CKPT_PATH", DEFAULT_TEXT_CKPT)
AUDIO_CKPT_PATH = os.getenv("AUDIO_CKPT_PATH", DEFAULT_AUDIO_CKPT)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#TOKENIZER = AutoTokenizer.from_pretrained("vinai/phobert-large")
TOKENIZER = BertTokenizer.from_pretrained('bert-large-uncased')
AUDIO_PROCESSOR = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-large")

TEXT_MODEL = PhoBERTEmbeddingModel(embedding_dim=1024, projection_dim=512).to(device)
AUDIO_MODEL = WavLMEmbeddingModel(embedding_dim=1024, projection_dim=512).to(device)


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

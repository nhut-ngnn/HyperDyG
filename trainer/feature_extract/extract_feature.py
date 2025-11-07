import os
import sys
import pickle
import warnings
import argparse

import numpy as np
import torch
import soundfile as sf
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.feature_extract.config import (
    PKL_DIR, OUTPUT_DIR, device,
    TOKENIZER, AUDIO_PROCESSOR,
    TEXT_MODEL, AUDIO_MODEL
)


def _load_audio(audio_path):
    array, sr = sf.read(audio_path)
    if isinstance(array, np.ndarray):
        waveform = array.astype(np.float32)
    else:
        waveform = np.array(array, dtype=np.float32)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    return np.ascontiguousarray(waveform), sr


def _extract_audio_embedding(waveform, sr, processor, model, device, source=None):
    try:
        inputs = processor(
            waveform,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            pooled, _ = model(inputs["input_values"])
        return pooled.squeeze().cpu()
    except Exception as e:
        prefix = f"{source}: " if source else ""
        print(f"[ERROR] Audio failed: {prefix}({e})")
        return None


def extract_audio_features(audio_path, processor, model, device):
    try:
        waveform, sr = _load_audio(audio_path)
    except Exception as e:
        print(f"[ERROR] Audio failed: {audio_path} ({e})")
        return None
    return _extract_audio_embedding(
        waveform,
        sr,
        processor,
        model,
        device,
        source=audio_path
    )

def extract_text_features(text, tokenizer, model, device):
    try:
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        ).to(device)
        with torch.no_grad():
            pooled, _ = model(inputs["input_ids"], inputs["attention_mask"]) 
        return pooled.squeeze().cpu()
    except Exception as e:
        print(f"[ERROR] Text failed: {text[:30]}... ({e})")
        return None

def process_single_sample(audio_path, text, label, is_pseudo=False, confidence=None, skip_text=False):
    audio_embed = extract_audio_features(audio_path, AUDIO_PROCESSOR, AUDIO_MODEL, device)
    if audio_embed is None:
        return None

    if skip_text:
        text_embed = torch.zeros_like(audio_embed)
    else:
        text_embed = extract_text_features(text, TOKENIZER, TEXT_MODEL, device)
        if text_embed is None:
            return None

    return {
        "text_embed": text_embed,
        "audio_embed": audio_embed,
        "label": label,
        "is_pseudo": is_pseudo,
        "confidence": confidence,
        "sample_id": os.path.basename(audio_path),
        "raw_text": text,
        "audio_path": audio_path
    }


def process_dataset(pkl_path, wav_base, output_path, pseudo=False, skip_text=False):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    processed_samples = []
    print(f"Processing {len(data)} samples from {pkl_path}")

    for item in tqdm(data, desc=f"Processing {os.path.basename(pkl_path)}"):
        if isinstance(item, dict):
            filename = item.get("filename") or item.get("audio_path") or item.get("path")
        elif isinstance(item, (tuple, list)):
            filename = item[0]
        else:
            raise TypeError(f"Unexpected item type: {type(item)}")

        if os.path.isabs(filename):
            audio_path = filename
        else:
            audio_path = os.path.join(wav_base, os.path.basename(filename))

        if isinstance(item, dict):
            text = item.get("text") or item.get("transcript") or item.get("utterance")
        else:
            text = item[1] if len(item) > 1 else ""

        label = None
        conf = None
        if isinstance(item, dict):
            if pseudo:
                label = item.get("pseudo_label", item.get("emotion"))
                conf = item.get("confidence", None)
            else:
                label = item.get("emotion")
                conf = item.get("confidence", None)
        elif isinstance(item, (tuple, list)) and len(item) > 2:
            label = item[2]

        sample = process_single_sample(
            audio_path,
            text,
            label,
            is_pseudo=pseudo,
            confidence=conf,
            skip_text=skip_text,
        )
        if sample is not None:
            processed_samples.append(sample)
        else:
            print(f"[SKIP] Failed to process: {audio_path}")

    with open(output_path, "wb") as f:
        pickle.dump(processed_samples, f)

    print(f"Saved processed data to: {output_path}")
    print(f"Total processed samples: {len(processed_samples)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["IEMOCAP", "ViSEC", "ESD", "ESD_MANDARIN"],
        help="Dataset to process",
    )
    parser.add_argument("--pseudo", action="store_true", help="Flag to process dataset as pseudo-labeled")
    parser.add_argument("--wav_base", type=str, default=None, help="Root directory containing the waveform files for the dataset.")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Starting feature extraction...")
    print(f"Using device: {device}")

    datasets = []
    if args.dataset == "IEMOCAP":
        pkl_prefix = "IEMOCAP"
        datasets = [
            ("train", f"{pkl_prefix}_preprocessed/train.pkl", f"{pkl_prefix}_BERT_WavLM_train.pkl", False),
            ("val",   f"{pkl_prefix}_preprocessed/val.pkl",   f"{pkl_prefix}_BERT_WavLM_val.pkl", False),
            ("test",  f"{pkl_prefix}_preprocessed/test.pkl",  f"{pkl_prefix}_BERT_WavLM_test.pkl", False),
        ]
    elif args.dataset == "ViSEC":
        pkl_prefix = "ViSEC"
        datasets = [
            ("train", f"{pkl_prefix}_preprocessed/train.pkl", f"{pkl_prefix}_BERT_WavLM_train.pkl", False),
            ("val",   f"{pkl_prefix}_preprocessed/val.pkl",   f"{pkl_prefix}_BERT_WavLM_val.pkl", False),
            ("test",  f"{pkl_prefix}_preprocessed/test.pkl",  f"{pkl_prefix}_BERT_WavLM_test.pkl", False),
        ]
    elif args.dataset == "ESD":
        pkl_prefix = "ESD"
        datasets = [
            ("train", f"{pkl_prefix}_preprocessed/train.pkl", f"{pkl_prefix}_BERT_WavLM_train.pkl", False),
            ("val",   f"{pkl_prefix}_preprocessed/val.pkl",   f"{pkl_prefix}_BERT_WavLM_val.pkl", False),
            ("test",  f"{pkl_prefix}_preprocessed/test.pkl",  f"{pkl_prefix}_BERT_WavLM_test.pkl", False),
        ]
    elif args.dataset == "ESD_MANDARIN":
        pkl_prefix = "ESD_mandarin"
        datasets = [
            ("test", f"{pkl_prefix}_preprocessed/test.pkl", f"{pkl_prefix}_WavLM_test.pkl", True),
        ]

    for split_name, pkl_file, output_file, skip_text in datasets:
        print(f"\n{'='*50}")
        print(f"Processing {split_name} split: {pkl_file}")
        print(f"{'='*50}")
        wav_base = args.wav_base
        if wav_base is None:
            raise ValueError("Please provide --wav_base to specify the audio root directory.")
        process_dataset(
            os.path.join(PKL_DIR, pkl_file),
            wav_base,
            os.path.join(OUTPUT_DIR, output_file),
            pseudo=args.pseudo if split_name == "train" else False,
            skip_text=skip_text,
        )

if __name__ == "__main__":
    main()

import argparse
import os
import pickle
import sys
import warnings
from collections import OrderedDict
from math import ceil

import torch
import torchaudio
from tqdm import tqdm
import numpy as np

warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


from src.feature_extract.config import ( 
    AUDIO_MODEL,
    AUDIO_PROCESSOR,
    OUTPUT_DIR,
    PKL_DIR,
    TEXT_MODEL,
    TOKENIZER,
    device,
)
from src.utils.utils import set_seed


DEFAULT_NOISE_FILENAMES = {
    "Babble": "babble.wav",
    "F16": "f16.wav",
    "Volvo": "volvo.wav",
    "HF-Channel": "hfchannel.wav",
    "White": "white.wav",
}
TARGET_SAMPLE_RATE = 16_000
ESD_TRANSCRIPT_CACHE = {}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract noisy test embeddings using shared BERT and WavLM encoders."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["IEMOCAP", "ESD"],
        help="Dataset to process.",
    )
    parser.add_argument(
        "--wav-base",
        type=str,
        default=None,
        help="Directory containing wav files referenced in the metadata.",
    )
    parser.add_argument(
        "--noise-dir",
        type=str,
        default="NoiseX-92",
        help="Directory that stores NoiseX-92 wav files.",
    )
    parser.add_argument(
        "--noise-types",
        type=str,
        default="Babble,F16,Volvo,HF-Channel,White",
        help="Comma-separated list of noise types to apply.",
    )
    parser.add_argument(
        "--snr-levels",
        type=str,
        default="20,15,10,0",
        help="Comma-separated list of SNR levels (dB).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help="Directory to store generated pickle files.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Optional override for the saved file prefix.",
    )
    parser.add_argument(
        "--text-ckpt",
        type=str,
        default=None,
        help="Optional checkpoint to reload BERT embeddings.",
    )
    parser.add_argument(
        "--audio-ckpt",
        type=str,
        default=None,
        help="Optional checkpoint to reload WavLM embeddings.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def load_checkpoint(model, checkpoint_path, device_type):
    checkpoint = torch.load(checkpoint_path, map_location=device_type)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    formatted = OrderedDict()
    is_parallel = isinstance(model, torch.nn.DataParallel)

    for key, value in state_dict.items():
        if is_parallel:
            if not key.startswith("module."):
                key = f"module.{key}"
        else:
            if key.startswith("module."):
                key = key[len("module.") :]
        formatted[key] = value

    missing, unexpected = model.load_state_dict(formatted, strict=False)
    if missing:
        print(f"[WARN] Missing keys while loading {checkpoint_path}: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys while loading {checkpoint_path}: {unexpected}")


def load_noise_waveforms(noise_dir, noise_types):
    library = {}
    for name in noise_types:
        filename = DEFAULT_NOISE_FILENAMES.get(name, f"{name.lower()}.wav")
        path = os.path.join(noise_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Noise file not found for '{name}': {path}")
        waveform, sample_rate = torchaudio.load(path)
        if sample_rate != TARGET_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE
            )
        library[name] = waveform
    return library


def _load_esd_transcripts_for_speaker(speaker_dir):
    cached = ESD_TRANSCRIPT_CACHE.get(speaker_dir)
    if cached is not None:
        return cached

    transcripts = {}
    speaker_id = os.path.basename(os.path.normpath(speaker_dir))
    candidate_files = [
        os.path.join(speaker_dir, f"{speaker_id}.txt"),
        os.path.join(speaker_dir, f"{speaker_id}.TXT"),
    ]
    transcript_path = next((path for path in candidate_files if os.path.isfile(path)), None)
    if transcript_path is None:
        ESD_TRANSCRIPT_CACHE[speaker_dir] = transcripts
        return transcripts

    def normalize_key(raw_key):
        key = raw_key.strip()
        if key.lower().endswith(".wav"):
            key = key[:-4]
        return key.lower()

    with open(transcript_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            key, text = None, None
            for sep in ("|", "\t"):
                if sep in line:
                    key, text = line.split(sep, 1)
                    break
            if key is None:
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    key, text = parts
            if key is None:
                continue
            transcripts[normalize_key(key)] = text.strip()

    ESD_TRANSCRIPT_CACHE[speaker_dir] = transcripts
    return transcripts


def _maybe_fill_esd_text(audio_path, current_text):
    if isinstance(current_text, str) and current_text.strip():
        return current_text
    speaker_dir = os.path.dirname(os.path.dirname(os.path.abspath(audio_path)))
    transcripts = _load_esd_transcripts_for_speaker(speaker_dir)
    utt_id = os.path.splitext(os.path.basename(audio_path))[0].lower()
    return transcripts.get(utt_id, current_text or "")


def add_noise(clean_waveform, noise_waveform, snr_db):
    if noise_waveform.size(1) < clean_waveform.size(1):
        repeat_factor = ceil(clean_waveform.size(1) / noise_waveform.size(1))
        noise_waveform = noise_waveform.repeat(1, repeat_factor)

    noise_waveform = noise_waveform[:, : clean_waveform.size(1)]

    signal_power = clean_waveform.pow(2).mean().clamp_min(1e-12)
    noise_power = noise_waveform.pow(2).mean().clamp_min(1e-12)
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    scaling_factor = torch.sqrt(target_noise_power / noise_power)

    noisy_waveform = clean_waveform + noise_waveform * scaling_factor
    return noisy_waveform.clamp(-1.0, 1.0)


def extract_text_features(text, tokenizer, model):
    if not isinstance(text, str) or not text.strip():
        return None
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    ).to(device)
    with torch.no_grad():
        pooled, _ = model(inputs["input_ids"], inputs["attention_mask"])
    return pooled.squeeze().cpu()


def extract_audio_features(audio_path, processor, model, noise_waveform, snr_db):
    if not os.path.isfile(audio_path):
        print(f"[WARN] Missing audio file: {audio_path}")
        return None

    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE
        )

    noisy_waveform = add_noise(waveform, noise_waveform, snr_db)
    sample = noisy_waveform.squeeze(0).detach().cpu().numpy().astype(np.float32)
    if sample.size == 0:
        print(f"[WARN] Empty waveform after noise addition: {audio_path}")
        return None

    try:
        inputs = processor(
            [sample],
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
    except ValueError:
        try:
            inputs = processor(
                sample,
                sampling_rate=TARGET_SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
            )
        except Exception as exc:
            print(f"[WARN] Processor failed for {audio_path}: {exc}")
            return None
    except Exception as exc:
        print(f"[WARN] Processor failed for {audio_path}: {exc}")
        return None
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        pooled, _ = model(inputs["input_values"])
    return pooled.squeeze().cpu()


def process_dataset(
    pkl_path,
    wav_base,
    noise_type,
    snr_db,
    tokenizer,
    text_model,
    processor,
    audio_model,
    noise_waveform,
    dataset,
):
    with open(pkl_path, "rb") as handle:
        data = pickle.load(handle)

    print(f"Processing {len(data)} samples from {pkl_path}")
    processed_samples = []

    iterator = tqdm(data, desc=f"{noise_type} {snr_db}dB")
    for item in iterator:
        filename = None
        text = ""
        label = None
        confidence = None
        is_pseudo = False

        if isinstance(item, dict):
            filename = (
                item.get("filename")
                or item.get("audio_path")
                or item.get("path")
                or item.get("wav")
            )
            text = (
                item.get("text")
                or item.get("transcript")
                or item.get("utterance")
                or ""
            )
            if "pseudo_label" in item and item["pseudo_label"] is not None:
                label = item["pseudo_label"]
                is_pseudo = True
            else:
                label = item.get("emotion") or item.get("label")
                is_pseudo = bool(item.get("is_pseudo", False))
            confidence = item.get("confidence")
        elif isinstance(item, (list, tuple)):
            if not item:
                print("[SKIP] Empty sample encountered.")
                continue
            filename = item[0]
            text = item[1] if len(item) > 1 else ""
            label = item[2] if len(item) > 2 else None
            confidence = item[3] if len(item) > 3 else None
        else:
            print(f"[SKIP] Unexpected item type: {type(item)}")
            continue

        if not isinstance(filename, str) or not filename:
            print(f"[SKIP] Missing filename in sample: {item}")
            continue

        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        if os.path.isabs(filename):
            audio_path = filename
        elif wav_base:
            audio_path = os.path.join(wav_base, filename)
        else:
            audio_path = filename
        audio_path = os.path.abspath(audio_path)

        if dataset == "ESD":
            text = _maybe_fill_esd_text(audio_path, text)

        text_embed = extract_text_features(text, tokenizer, text_model)
        audio_embed = extract_audio_features(
            audio_path,
            processor,
            audio_model,
            noise_waveform,
            snr_db,
        )

        if text_embed is None or audio_embed is None:
            print(f"[SKIP] Failed to process {audio_path}")
            continue

        processed_samples.append(
            {
                "text_embed": text_embed,
                "audio_embed": audio_embed,
                "label": label,
                "is_pseudo": is_pseudo,
                "confidence": confidence,
                "sample_id": os.path.basename(audio_path),
                "raw_text": text,
                "audio_path": audio_path,
            }
        )

    return processed_samples


def main():
    args = parse_args()
    set_seed(args.seed)

    text_model = TEXT_MODEL
    audio_model = AUDIO_MODEL
    text_model.eval()
    audio_model.eval()

    if args.text_ckpt:
        print(f"Loading text checkpoint from {args.text_ckpt}")
        load_checkpoint(text_model, args.text_ckpt, device)
    if args.audio_ckpt:
        print(f"Loading audio checkpoint from {args.audio_ckpt}")
        load_checkpoint(audio_model, args.audio_ckpt, device)

    if args.dataset == "IEMOCAP":
        pkl_prefix = "IEMOCAP"
    elif args.dataset == "ESD":
        pkl_prefix = "ESD"
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    test_pkl = os.path.join(PKL_DIR, f"{pkl_prefix}_preprocessed/test.pkl")
    if not os.path.isfile(test_pkl):
        raise FileNotFoundError(f"Test metadata not found: {test_pkl}")

    output_dir = args.output_dir or OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    noise_types = [name.strip() for name in args.noise_types.split(",") if name.strip()]
    snr_levels = [float(val.strip()) for val in args.snr_levels.split(",") if val.strip()]
    noise_library = load_noise_waveforms(args.noise_dir, noise_types)

    default_prefix = f"{pkl_prefix}_BERT_WavLM_test"
    output_prefix = args.output_prefix or default_prefix

    for snr_db in snr_levels:
        for noise_type in noise_types:
            print(f"\nProcessing noise='{noise_type}' at {snr_db} dB SNR")
            processed = process_dataset(
                pkl_path=test_pkl,
                wav_base=args.wav_base,
                noise_type=noise_type,
                snr_db=snr_db,
                tokenizer=TOKENIZER,
                text_model=text_model,
                processor=AUDIO_PROCESSOR,
                audio_model=audio_model,
                noise_waveform=noise_library[noise_type],
                dataset=args.dataset,
            )

            snr_suffix = f"{int(snr_db) if snr_db.is_integer() else snr_db}dB"
            filename = f"{output_prefix}_{noise_type.replace('-', '_')}_{snr_suffix}.pkl"
            output_path = os.path.join(output_dir, filename)

            with open(output_path, "wb") as handle:
                pickle.dump(processed, handle)

            print(f"Saved {len(processed)} samples to {output_path}")


if __name__ == "__main__":
    main()

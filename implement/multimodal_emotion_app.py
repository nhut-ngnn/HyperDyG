import os
import io
import torch
import torchaudio
import streamlit as st
import numpy as np
import pickle
try:
    import sounddevice as sd
    import wavio
    SOUNDDEVICE_AVAILABLE = True
    SOUNDDEVICE_ERR = None
except Exception as _e:
    sd = None 
    wavio = None 
    SOUNDDEVICE_AVAILABLE = False
    SOUNDDEVICE_ERR = str(_e)

try:
    from streamlit_mic_recorder import mic_recorder
    MIC_WEB_AVAILABLE = True
except Exception:
    MIC_WEB_AVAILABLE = False
import soundfile as sf
from transformers import (
    AutoFeatureExtractor,
    AutoTokenizer,
    BertTokenizer,
    Wav2Vec2Processor,
    AutoProcessor,
    Wav2Vec2ForCTC,
    AutoModelForSpeechSeq2Seq,
)
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    HAS_ACCELERATE = True
except Exception:
    HAS_ACCELERATE = False

from src.feature_extract.model_encode import (
    BERTEmbeddingModel,
    PhoBERTEmbeddingModel,
    WavLMEmbeddingModel
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INFER_DIR = os.path.join(BASE_DIR, "inference_file")
os.makedirs(INFER_DIR, exist_ok=True)
AUDIO_PATH = os.path.join(INFER_DIR, "recorded.wav")
TRANSCRIPT_PATH = os.path.join(INFER_DIR, "transcript.txt")
OUTPUT_PKL = os.path.join(INFER_DIR, "sample_embeddings.pkl")
def get_best_device(min_free_gb: float = 2.0) -> str:
    """Choose cuda only if available and has at least `min_free_gb` free; else cpu."""
    if torch.cuda.is_available():
        try:
            free, total = torch.cuda.mem_get_info()  # bytes
            if free / (1024**3) >= min_free_gb:
                return "cuda"
        except Exception:
            return "cuda"
    return "cpu"

DEVICE = get_best_device()

try:
    from pyngrok import ngrok, conf as ngrok_conf
    NGROK_AVAILABLE = True
except Exception:
    NGROK_AVAILABLE = False

def record_audio(duration=5, fs=16000, output_file=AUDIO_PATH):
    if not SOUNDDEVICE_AVAILABLE:
        raise RuntimeError(
            "Recording is unavailable: PortAudio/sounddevice not installed. "
            "Please install system PortAudio and the sounddevice package, or upload an audio file instead."
        )
    st.write("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    wavio.write(output_file, audio, fs, sampwidth=2)
    st.write(f"Audio saved to {output_file}")
    return output_file

def _save_wav_bytes(wav_bytes: bytes, path: str):
    with open(path, "wb") as f:
        f.write(wav_bytes)
    return path

def reduce_noise(audio_tensor, sample_rate=16000):
    from torchaudio.transforms import Spectrogram, GriffinLim
    spec_transform = Spectrogram(n_fft=512, power=None)
    spec = spec_transform(audio_tensor)
    mag, phase = torch.abs(spec), torch.angle(spec)
    noise_threshold = mag.mean() * 0.5
    mag_denoised = torch.where(mag > noise_threshold, mag, torch.tensor(0.0, device=mag.device))
    spec_denoised = mag_denoised * torch.exp(1j * phase)
    griffin = GriffinLim(n_fft=512)
    clean_audio = griffin(torch.abs(spec_denoised))
    return clean_audio


def _guess_streamlit_port(default: int = 8501) -> int:
    for key in ["STREAMLIT_SERVER_PORT", "PORT", "SERVER_PORT"]:
        val = os.environ.get(key)
        if val and str(val).isdigit():
            return int(val)
    return default


@st.cache_resource
def start_ngrok(port: int, token: str | None = None):
    if not NGROK_AVAILABLE:
        raise RuntimeError("pyngrok is not installed. Install with: pip install pyngrok")
    if token:
        try:
            ngrok_conf.get_default().auth_token = token
        except Exception:
            pass
    try:
        existing = ngrok.get_tunnels()
        for t in existing:
            try:
                cfg = getattr(t, 'config', None) or {}
                addr = cfg.get('addr', '') if isinstance(cfg, dict) else ''
                if addr.endswith(f":{port}"):
                    return t.public_url
            except Exception:
                continue
    except Exception:
        pass
    public_url = ngrok.connect(port, "http").public_url
    return public_url

@st.cache_resource
def load_wav2vec2_s2t_model(device: str = "cpu"):
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-large-960h",
        torch_dtype=dtype,
        low_cpu_mem_usage=HAS_ACCELERATE,
    ).to(device)
    model.eval()
    return processor, model

@st.cache_resource
def load_phowhisper_model(device: str = "cpu"):
    model_name = "vinai/PhoWhisper-small"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=HAS_ACCELERATE,
    ).to(device)
    model.eval()
    return processor, model

def transcribe_en(audio_path, processor, model, output_txt_path, device: str = "cpu"):
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    waveform = reduce_noise(waveform)
    input_values = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").input_values.to(device)
    with torch.inference_mode():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(transcription.strip())
    return transcription

def transcribe_vi(audio_path, processor, model, output_txt_path, device: str = "cpu"):
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    waveform = reduce_noise(waveform)
    inputs = processor(audio=waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").to(device)
    with torch.inference_mode():
        generated_ids = model.generate(**inputs)
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(transcription.strip())
    return transcription

def load_embedding_models(dataset, device: str = "cpu"):
    if dataset == "ViSEC":
        text_ckpt = os.path.join(BASE_DIR, f"model/EmbeddingModel/ViSEC/models/best_phobert_embeddings.pt")
        audio_ckpt = os.path.join(BASE_DIR, f"model/EmbeddingModel/ViSEC/models/best_wavlm_embeddings.pt")
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-large")
        text_model = PhoBERTEmbeddingModel(embedding_dim=1024, projection_dim=512).to(device)
    else:
        text_ckpt = os.path.join(BASE_DIR, f"model/EmbeddingModel/{dataset}/models/best_bert_embeddings.pt")
        audio_ckpt = os.path.join(BASE_DIR, f"model/EmbeddingModel/{dataset}/models/best_wavlm_embeddings.pt")
        tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
        text_model = BERTEmbeddingModel(embedding_dim=1024, projection_dim=512).to(device)

    processor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-large")
    audio_model = WavLMEmbeddingModel(embedding_dim=1024, projection_dim=512).to(device)

    text_model.load_state_dict(torch.load(text_ckpt, map_location=device)["model_state_dict"])
    audio_model.load_state_dict(torch.load(audio_ckpt, map_location=device)["model_state_dict"])

    text_model.eval()
    audio_model.eval()
    return tokenizer, processor, text_model, audio_model

def extract_text_features(text, tokenizer, model, device: str = "cpu"):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        pooled, _ = model(inputs["input_ids"], inputs["attention_mask"])
    return pooled.squeeze().cpu()

def extract_audio_features(audio_path, processor, model, device: str = "cpu"):
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
    input_values = inputs.input_values.to(device)
    with torch.inference_mode():
        pooled, _ = model(input_values)
    return pooled.squeeze().cpu()

def process_and_save(dataset, device: str = "cpu"):
    tokenizer, processor, text_model, audio_model = load_embedding_models(dataset, device)
    with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
        text = f.read().strip()
    text_embed = extract_text_features(text, tokenizer, text_model, device)
    audio_embed = extract_audio_features(AUDIO_PATH, processor, audio_model, device)
    sample = {
        "text_embed": text_embed,
        "audio_embed": audio_embed,
        "label": None,
        "valence_label": None,
        "arousal_label": None,
        "is_augmented": False,
        "sample_id": "audio_sample"
    }
    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump([sample], f)
    st.success(f"Feature saved to {OUTPUT_PKL}")

def main():
    st.set_page_config(page_title="Speech-to-Text + Embedding Extractor", layout="wide")
    st.title("Speech-to-Text + Embedding Extraction")

    # Sidebar: ngrok public link
    with st.sidebar:
        st.subheader("Public link (ngrok)")
        port_default = _guess_streamlit_port()
        port = st.number_input("Local port", min_value=1, max_value=65535, value=port_default, step=1)
        token_env = os.environ.get("NGROK_AUTHTOKEN") or os.environ.get("NGROK_TOKEN")
        token = st.text_input("ngrok auth token (optional)", value=token_env or "", type="password")
        force_cpu = st.toggle("Force CPU (avoid GPU OOM)", value=False, help="Run all models on CPU to prevent CUDA OOM")
        st.session_state["force_cpu"] = force_cpu
        current_device = "cpu" if force_cpu else get_best_device()
        st.caption(f"Using device: {current_device}")
        if st.button("Start ngrok"):
            try:
                url = start_ngrok(int(port), token if token else None)
                st.success(f"Public URL: {url}")
                st.write("Share this URL with clients to access the app.")
            except Exception as e:
                st.error(str(e))
        if NGROK_AVAILABLE:
            try:
                # Show current tunnels
                tunnels = [t.public_url for t in ngrok.get_tunnels()]
                if tunnels:
                    st.caption("Active tunnels:")
                    for u in tunnels:
                        st.code(u)
            except Exception:
                pass

    lang = st.radio("Select Language", ["English", "Vietnamese"], horizontal=True)

    if lang == "Vietnamese":
        dataset = "ViSEC"
    else:
        dataset = st.radio("Select Dataset", ["IEMOCAP", "ESD"], horizontal=True)

    st.markdown("### Record from browser (client microphone)")
    if MIC_WEB_AVAILABLE:
        rec = mic_recorder(
            start_prompt="Start recording",
            stop_prompt="Stop",
            key="mic_web",
            just_once=False,
            format="wav",
        )
        if rec and isinstance(rec, dict) and rec.get("bytes"):
            audio_path = _save_wav_bytes(rec["bytes"], AUDIO_PATH)
            st.session_state["audio_path"] = audio_path
            st.success("Captured from browser mic.")
            st.audio(audio_path, format="audio/wav")
    else:
        st.info("Browser mic recorder not installed. Install with: pip install streamlit-mic-recorder")

    # Server-side recording removed per request; use browser mic or upload instead.

    uploaded = st.file_uploader("Or upload your audio (.wav, .mp3)", type=["wav", "mp3"])
    if uploaded:
        audio_path = os.path.join(INFER_DIR, "uploaded.wav")
        with open(audio_path, "wb") as f:
            f.write(uploaded.read())
        st.audio(audio_path, format="audio/wav")
        st.session_state["audio_path"] = audio_path

    if "audio_path" in st.session_state and st.button("Run Speech-to-Text"):
        output_txt_path = TRANSCRIPT_PATH
        with st.spinner("Transcribing..."):
            device = "cpu" if st.session_state.get("force_cpu") else get_best_device()
            if lang == "English":
                processor, model = load_wav2vec2_s2t_model(device)
                transcription = transcribe_en(st.session_state["audio_path"], processor, model, output_txt_path, device)
            else:
                processor, model = load_phowhisper_model(device)
                transcription = transcribe_vi(st.session_state["audio_path"], processor, model, output_txt_path, device)
        st.text_area("Transcript", transcription, height=200)
        st.session_state["transcribed"] = True

    st.subheader("Model Comparison")
    selected_models = st.multiselect(
        "Select Models to Evaluate",
        ["FleSER", "MemoCMT", "GloMER", "HyperDyG"],
        default=["FleSER", "GloMER"]
    )

    if st.button("Extract Features & Run Inference"):
        if not st.session_state.get("audio_path"):
            st.warning("Please record or upload audio first.")
        elif not st.session_state.get("transcribed"):
            st.warning("Please run Speech-to-Text first to create transcript.")
        elif not selected_models:
            st.warning("Please select at least one model to evaluate.")
        else:
            device = "cpu" if st.session_state.get("force_cpu") else get_best_device()
            if device == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            with st.spinner("Extracting embeddings..."):
                process_and_save(dataset, device)

            with open(OUTPUT_PKL, "rb") as f:
                features = pickle.load(f)[0]

            results = []
            for model_name in selected_models:
                model_path = os.path.join(
                    BASE_DIR, f"model/EmotionModel/{dataset}/{model_name}/{dataset}_{model_name}.pt"
                )
                if not os.path.exists(model_path):
                    results.append({
                        "Model": model_name,
                        "Prediction": "(missing checkpoint)",
                        "Time (ms)": None
                    })
                    continue

                try:
                    run_device = "cpu" if st.session_state.get("force_cpu") else get_best_device()
                    model = load_model(model_name, model_path, run_device, dataset)
                    model.eval()
                    import time
                    start = time.time()
                    with torch.inference_mode():
                        text_in = features["text_embed"].unsqueeze(0).to(run_device)
                        audio_in = features["audio_embed"].unsqueeze(0).to(run_device)
                        if model_name == "GloMER":
                            pred = model(text_in, audio_in, return_cls=True)
                        else:
                            pred = model(text_in, audio_in)
                        if isinstance(pred, tuple):
                            pred = pred[0]
                    elapsed = (time.time() - start) * 1000
                    label = get_label_name(pred, dataset)
                    results.append({
                        "Model": model_name,
                        "Prediction": label,
                        "Time (ms)": round(elapsed, 2)
                    })
                except torch.cuda.OutOfMemoryError as e:
                    # Fallback to CPU for this model
                    try:
                        st.warning(f"GPU OOM for {model_name}, falling back to CPU…")
                        model = load_model(model_name, model_path, "cpu", dataset)
                        model.eval()
                        import time
                        start = time.time()
                        with torch.inference_mode():
                            text_in = features["text_embed"].unsqueeze(0)
                            audio_in = features["audio_embed"].unsqueeze(0)
                            if model_name == "GloMER":
                                pred = model(text_in, audio_in, return_cls=True)
                            else:
                                pred = model(text_in, audio_in)
                            if isinstance(pred, tuple):
                                pred = pred[0]
                        elapsed = (time.time() - start) * 1000
                        label = get_label_name(pred, dataset)
                        results.append({
                            "Model": model_name,
                            "Prediction": label,
                            "Time (ms)": round(elapsed, 2)
                        })
                    except Exception as ee:
                        results.append({
                            "Model": model_name,
                            "Prediction": f"Error after CPU fallback: {str(ee)}",
                            "Time (ms)": None
                        })
                except Exception as e:
                    results.append({
                        "Model": model_name,
                        "Prediction": f"Error: {str(e)}",
                        "Time (ms)": None
                    })

            if results:
                try:
                    import pandas as pd
                    df = pd.DataFrame(results)
                    st.dataframe(df, use_container_width=True)
                except Exception:
                    st.write("Results:")
                    for r in results:
                        st.write(f"- {r['Model']}: {r['Prediction']} ({r['Time (ms)']} ms)")

def load_model(name, path, device, dataset):
    import torch
    state_dict = torch.load(path, map_location=device)
    sd = state_dict.get('model_state_dict', state_dict)
    def _infer_dim(key_candidates, d, default=1024):
        try:
            for k in key_candidates:
                for state_k, tensor in d.items():
                    if k in state_k and hasattr(tensor, 'shape') and len(tensor.shape) > 1:
                        return tensor.shape[1]
        except Exception:
            pass
        return default
    def _infer_num_classes(d, default=4):
        try:
            candidates = []
            for k, tensor in d.items():
                if not hasattr(tensor, 'shape'):
                    continue
                if 'classifier' in k and k.endswith('.weight') and len(tensor.shape) == 2:
                    candidates.append(tensor.shape[0])
            if candidates:
                return min(candidates)
        except Exception:
            pass
        return default
    inferred_text_dim = _infer_dim(['text', 'bert', 'phobert'], sd)
    inferred_audio_dim = _infer_dim(['audio', 'wavlm'], sd)
    dataset_num_classes = {"IEMOCAP": 4, "ESD": 5, "ViSEC": 4}
    inferred_num_classes = _infer_num_classes(sd)
    num_classes = dataset_num_classes.get(dataset, inferred_num_classes)
    def _infer_hyperdyg_dims(d):
        try:
            for k, tensor in d.items():
                if 'fusion' in k and hasattr(tensor, 'shape') and len(tensor.shape) > 1:
                    return tensor.shape[1]
        except Exception:
            pass
        return None
    if name == "GloMER":
        from src.GloMER.Model import CrossModalContrastiveModel
        model = CrossModalContrastiveModel(
            text_input_dim=inferred_text_dim,
            audio_input_dim=inferred_audio_dim,
            fusion_dim=512,
            projection_dim=256,
            num_heads=8,
            dropout=0.2,
            linear_layer_dims=[512, 128],
            num_classes=num_classes
        ).to(device)
    elif name == "MemoCMT":
        from src.MemoCMT.Model import MemoCMT, Config as MemoConfig
        memo_cfg = MemoConfig(
            text_encoder_dim=inferred_text_dim,
            audio_encoder_dim=inferred_audio_dim,
            fusion_dim=512,
            num_attention_head=8,
            dropout=0.2,
            linear_layer_output=[512, 128],
            num_classes=num_classes,
            fusion_head_output_type="mean",
        )
        model = MemoCMT(memo_cfg, device=device).to(device)
    elif name == "FleSER":
        from src.FleSER.Model import FlexibleMMSER
        model = FlexibleMMSER(
            text_input_dim=inferred_text_dim,
            audio_input_dim=inferred_audio_dim,
            num_classes=num_classes,
            fusion_method='self_attention',
            alpha=0.5,
            dropout_rate=0.2,
            use_layernorm=True,
            text_only=False,
            hidden_dim=512,
            proj_dim=256,
            num_heads=8
        ).to(device)
    elif name == "HyperDyG":
        from src.HyperDyG.Model import HyperDyG
        inferred_fusion = _infer_hyperdyg_dims(sd)
        fusion_dim_to_use = inferred_fusion if inferred_fusion is not None else 512
        model = HyperDyG(
            text_input_dim=inferred_text_dim,
            audio_input_dim=inferred_audio_dim,
            fusion_dim=fusion_dim_to_use,
            projection_dim=256,
            num_heads=8,
            dropout=0.2,
            linear_layer_dims=[512, 128],
            num_classes=num_classes,
            hypergraph_k_text=5,
            hypergraph_k_audio=5,
            hypergraph_threshold=0.5,
            use_rgcn=True,
            use_gtr=True,
            num_relations=3
        ).to(device)
    else:
        raise ValueError(f"Unknown model name: {name}")
    model.load_state_dict(sd, strict=False)
    model.to(device)
    return model

def get_label_name(pred, dataset):
    labels = {
        "IEMOCAP": ["Angry", "Happy", "Sad", "Neutral"],
        "ESD": ["Angry", "Happy", "Sad", "Neutral", "Surprise"],
        "ViSEC": ["Happy", "Neutral", "Sad", "Angry"]
    }
    return labels[dataset][torch.argmax(pred).item()]

if __name__ == "__main__":
    main()


# HyperDyG: Multimodal Emotion Recognition with Hypergraph Construction and Dynamic Gating Fusion

HyperDyG is a multimodal speech emotion recognition framework that jointly models textual and acoustic cues through hypergraph-based fusion and dynamic gating. The repository provides end-to-end utilities covering feature extraction, supervised and semi-supervised training, fine-tuning, evaluation, and an interactive Streamlit demo for both English and Vietnamese speech.

## Highlights
- **Multiple backbones**: Train HyperDyG alongside comparative architectures (GloMER, MemoCMT, FleSER) via the same trainer CLI.
- **Semi-supervised learning**: Mix labeled and pseudo-labeled utterances with configurable weak/strong augmentations.
- **Feature pipeline**: Reuse pre-computed PKL features or regenerate embeddings with the supplied extraction scripts.
- **Interactive demo**: Launch a Streamlit app with optional ngrok tunneling to collect recordings from client microphones.
- **Reproducible experiments**: Deterministic seeds, stratified splits, and CSV summaries for all metrics.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `features_output/` | Serialized audio/text embeddings for IEMOCAP, ESD, ViSEC, etc. |
| `script/` | Shell wrappers for preprocessing, feature extraction, fine-tuning, and training. |
| `src/` | Core model architectures, projection heads, and embedding encoders. |
| `trainer/` | Training, semi-supervised training, prediction, preprocessing pipelines. |
| `implement/multimodal_emotion_app.py` | Streamlit demo for ASR, embedding extraction, and model comparison. |
| `requirements.txt` | Python dependencies. |

## Quick Start

1. **Create environment**
	 ```bash
	 python3 -m venv .venv
	 source .venv/bin/activate
	 pip install -r requirements.txt
	 ```

2. **Check GPU availability** (optional)
	 ```bash
	 nvidia-smi
	 ```

3. **Prepare feature pickles**
	- Use the provided PKL files under `features_output/`, or
	- Re-generate via `script/feature_extract.sh` after configuring paths in `src/feature_extract/config.py`.

4. **Run training via script**
	- Edit `script/train_semi.sh` to point to your dataset PKLs, model type, and hyperparameters.
	- Launch training with:
	  ```bash
	  bash script/train_semi.sh
	  ```
	- The script delegates to `trainer/train_semi.py`; adjust that file if custom behavior is required.

## Training Pipelines

### Supervised and Semi-Supervised Training
The `script/train_semi.sh` wrapper sets up environment variables and then calls `trainer/train_semi.py` with your chosen configuration. Key options to review inside the script or when invoking the trainer directly include:
- `--model_type {GloMER,MemoCMT,FleSER,HyperDyG}` — select architecture.
- `--unlabeled_ratio` — portion of training data treated as unlabeled for semi-supervised runs.
- Augmentation flags such as `--disable_augmentation` or `--supervised_strong_aug`.

Both supervised-only and semi-supervised phases compute weighted cross-entropy, track WA/UA/WF1/UF1, and emit per-seed CSV logs in `results/`.

## Evaluation and Prediction

Use `trainer/predict.py` to evaluate saved checkpoints:

```bash
source .venv/bin/activate
python trainer/predict.py \
	--config configs/eval/iemocap.yaml \
	--checkpoint runs/hyperdyg/best_model.pt
```

Metrics mirror the training stage; results can be exported to CSV or JSON for analysis.

## Feature Extraction and Fine-Tuning

- **Speech/text encoders**: `src/feature_extract/model_encode.py` defines BERT/PhoBERT and WavLM heads.
- **Fine-tuning**: Adjust backbone checkpoints via `script/fine_tuning.sh` or modules under `fine_tuning/`.
- **On-demand embeddings**: `trainer/extract_feature.py` regenerates PKLs from raw audio/transcript pairs.

## Streamlit Demo and ngrok Sharing

The Streamlit app supports browser-based recording, ASR (English Wav2Vec2, Vietnamese PhoWhisper), embedding extraction, and model comparison.

### Run locally
```bash
source .venv/bin/activate
streamlit run implement/multimodal_emotion_app.py \
	--server.port 8501 --server.address 0.0.0.0
```

### Expose publicly with ngrok
```bash
source .venv/bin/activate
ngrok config add-authtoken YOUR_TOKEN      # once per machine
ngrok http 8501
```

Copy the forwarding URL and share it with remote users. The app also includes an in-app “Start ngrok” button (requires `pyngrok`).

## 📚 Citation
If you use this repository, please cite the associated HyperDyG work (update with final bibliographic entry):

```
```

## Contact
For questions or collaboration requests, reach out to:

Nhut Minh Nguyen — FPT University, Vietnam

- **Email:** [minhnhut.ngnn@gmail.com](mailto:minhnhut.ngnn@gmail.com)
- **ORCID:** [https://orcid.org/0009-0003-1281-5346](https://orcid.org/0009-0003-1281-5346)
- **GitHub:** [https://github.com/nhut-ngnn](https://github.com/nhut-ngnn)

---

© 2025 Nhut Minh Nguyen. Licensed under the terms specified in `LICENSE`.

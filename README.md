# HyperDyG: Hypergraph-Driven Dynamic Fusion for Semi-Supervised Multimodal Emotion Recognition
<i>
  Official code repository for the manuscript 
  <b>"HyperDyG: Hypergraph-Driven Dynamic Fusion for Semi-Supervised Multimodal Emotion Recognition"</b>, 
  submitted at 
  <a href="https://publications.eai.eu/index.php/inis">EAI Endorsed Transactions on Industrial Networks and Intelligent Systems Journal</a>.
</i>

> Please press ⭐ button and/or cite papers if you feel helpful.

<p align="center">
<img src="https://img.shields.io/github/stars/nhut-ngnn/HyperDyG">
<img src="https://img.shields.io/github/forks/nhut-ngnn/HyperDyG">
<img src="https://img.shields.io/github/watchers/nhut-ngnn/HyperDyG">
</p>

<div align="center">

[![python](https://img.shields.io/badge/-Python_3.8.20-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![pytorch](https://img.shields.io/badge/Torch_2.0.1-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![cuda](https://img.shields.io/badge/-CUDA_11.8-green?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit-archive)
</div>

<p align="center">
<img src="https://img.shields.io/badge/Last%20updated%20on-18.11.2025-brightgreen?style=for-the-badge">
<img src="https://img.shields.io/badge/Written%20by-Nguyen%20Minh%20Nhut-pink?style=for-the-badge"> 
</p>


<div align="center">

[**Highlights**](#highlights) •
[**Repository Layout**](#repository-layout) •
[**Quick Start**](#quick-start) •
[**Training Pipelines**](#training-pipelines) •
[**References**](#references) •
[**Citation**](#citation) •
[**Contact**](#Contact)

</div>

## Highlights
- **Multiple backbones**: Train HyperDyG alongside comparative architectures (GloMER, MemoCMT, FleSER) via the same trainer CLI.
- **Semi-supervised learning**: Mix labeled and pseudo-labeled utterances with configurable weak/strong augmentations.
- **Feature pipeline**: Reuse pre-computed PKL features or regenerate embeddings with the supplied extraction scripts.
- **Interactive demo**: Launch a Streamlit app with optional ngrok tunneling to collect recordings from client microphones.
- **Reproducible experiments**: Deterministic seeds, stratified splits, and CSV summaries for all metrics.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `features_output/` | Serialized audio/text embeddings for IEMOCAP and ESD. |
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

### Evaluation and Prediction

Use `trainer/predict.py` to evaluate saved checkpoints:

```bash
source .venv/bin/activate
python trainer/predict.py \
	--config configs/eval/iemocap.yaml \
	--checkpoint runs/hyperdyg/best_model.pt
```

Metrics mirror the training stage; results can be exported to CSV or JSON for analysis.

### Feature Extraction and Fine-Tuning

- **Speech/text encoders**: `src/feature_extract/model_encode.py` defines BERT/PhoBERT and WavLM heads.
- **Fine-tuning**: Adjust backbone checkpoints via `script/fine_tuning.sh` or modules under `fine_tuning/`.
- **On-demand embeddings**: `trainer/extract_feature.py` regenerates PKLs from raw audio/transcript pairs.

## References
[1] Nhat Truong Pham, SERVER: Multi-modal Speech Emotion Recognition using Transformer-based and Vision-based Embeddings (ICIIT), 2023. Available https://github.com/nhattruongpham/mmser.git.

[2] Mustaqeem Khan, MemoCMT: Cross-Modal Transformer-Based Multimodal Emotion Recognition System (Scientific Reports), 2025. Available https://github.com/tpnam0901/MemoCMT.

[3] Nhut Minh Nguyen, HemoGAT: Heterogeneous multi-modal emotion recognition with cross-modal transformer and graph attention network, 2025. Available https://github.com/nhut-ngnn/HemoGAT.

[4] Nhut Minh Nguyen, GloMER: Towards Robust Multimodal Emotion Recognition via Gated Fusion and Contrastive Learning, 2025. Available https://github.com/nhut-ngnn/GloMER.

## Citation
If you use this code or part of it, please cite the following papers:
```
```
## Contact
For any information, please contact the main author:

Nhut Minh Nguyen at FPT University, Vietnam

**Email:** [minhnhut.ngnn@gmail.com](mailto:minhnhut.ngnn@gmail.com)<br>
**Website:** [https://nhut-ngnn.github.io/](https://nhut-ngnn.github.io/)<br>
**ORCID:** <link>https://orcid.org/0009-0003-1281-5346</link> <br>
**GitHub:** <link>https://github.com/nhut-ngnn/</link>
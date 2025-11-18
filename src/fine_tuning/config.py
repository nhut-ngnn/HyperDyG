
BERT_CONFIG = {
    "pkl_path": "metadata/IEMOCAP_preprocessed/train.pkl",
    "save_path": "fine_tuning/IEMOCAP/models/best_bert_embeddings.pt",
    "epochs": 10,
    "batch_size": 64,
    "lr": 2e-5,
    "max_length": 128,
    "embedding_dim": 768,
    "projection_dim": 512,
    "temperature": 0.07,
}

WAV_CONFIG = {
    "metadata_path": "metadata/ESD_preprocessed/train.pkl",
    "save_path": "fine_tuning/ESD/models/best_wavlm_embeddings.pt",
    "audio_dir": "/path/to/ESD/audio",
    "segment_length": 16000,
    "embedding_dim": 768,
    "projection_dim": 512,
    "batch_size": 64,
    "num_epochs": 10,
    "learning_rate": 1e-4,
    "temperature": 0.07,
    "seed": 42,
}

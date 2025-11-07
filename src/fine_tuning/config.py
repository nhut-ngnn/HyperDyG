
BERT_CONFIG = {
    "pkl_path": "/home/tri.pm/polyp/fptu/MinhNhut/HyperDyG_DSP_backup/metadata/ESD_preprocessed/train.pkl",
    "save_path": "/home/tri.pm/polyp/fptu/MinhNhut/fine_tuning/ESD/models/best_bert_embeddings.pt",
    "epochs": 5,
    "batch_size": 16,
    "lr": 2e-5,
    "max_length": 128,
    "embedding_dim": 1024,
    "projection_dim": 512,
    "temperature": 0.07,
}

WAV_CONFIG = {
    "metadata_path": "/home/tri.pm/polyp/fptu/MinhNhut/HyperDyG_DSP_backup/metadata/ESD_preprocessed/train.pkl",
    "save_path": "/home/tri.pm/polyp/fptu/MinhNhut/fine_tuning/ESD/models/best_wavlm_embeddings.pt",
    "audio_dir": "/home/tri.pm/polyp/fptu/MinhNhut/Emotion Speech Dataset/",
    "segment_length": 16000,
    "embedding_dim": 1024,
    "projection_dim": 512,
    "batch_size": 16,
    "num_epochs": 5,
    "learning_rate": 1e-4,
    "temperature": 0.07,
    "seed": 42,
}
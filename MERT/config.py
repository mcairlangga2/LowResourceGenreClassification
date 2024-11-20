#!/usr/bin/env python3
"""Configuration settings for MERT models."""

# Model configurations
MERT_CONFIGS = {
    "95m": {
        "name": "m-a-p/MERT-v1-95M",
        "embedding_size": 768,
        "default_lr": 2e-5,
        "default_batch_size": 8,
        "default_epochs": 5
    },
    "330m": {
        "name": "m-a-p/MERT-v1-330M",
        "embedding_size": 1024,
        "default_lr": 2e-5,
        "default_batch_size": 8,
        "default_epochs": 5
    }
}

# Training configurations
TRAINING_CONFIG = {
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "early_stopping_patience": 50,
    "early_stopping_threshold": 0.01,
    "save_steps": 15,
    "eval_steps": 15,
    "logging_steps": 15,
    "save_total_limit": 2
}

# Dataset configurations
DATASET_CONFIG = {
    "input_length": 240000,
    "sampling_rate": 24000,
    "num_classes": 9,
    "genre_mapping": {
        "Campursari": 0,
        "Dangdut": 1,
        "Keroncong": 2,
        "Baila": 3,
        "Wannam": 4,
        "Virindu": 5,
        "Qawwali": 6,
        "Pop": 7,
        "Ghazal": 8
    }
}

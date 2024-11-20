#!/usr/bin/env python3
"""
MERT Model Implementation for Genre Classification
Includes model architecture and dataset handling.
"""

import os
import torch
from torch import nn
from torch.utils import data
import numpy as np
import librosa
import random
from typing import Optional, Dict
from transformers import Wav2Vec2FeatureExtractor, AutoModel
from config import DATASET_CONFIG

class GTZANDataset(data.Dataset):
    """Dataset class for genre classification using MERT."""
    
    def __init__(self, root: str, split: str, processor: Wav2Vec2FeatureExtractor):
        """
        Initialize the dataset.
        
        Args:
            root: Root directory containing the dataset
            split: Dataset split ('train', 'valid', or 'test')
            processor: MERT feature processor
        """
        self.split = split.lower()
        self.mapping = DATASET_CONFIG['genre_mapping']
        self.files = [
            f for f in open(f"{root}/{split}_filtered.txt", "r").readlines()
            if "jazz.00054" not in f
        ]
        self.class_num = DATASET_CONFIG['num_classes']
        self.seg_length = DATASET_CONFIG['input_length']
        self.root = root
        self.processor = processor
        self.reverse_mapping = {v: k for k, v in self.mapping.items()}
        self.sampling_rate = DATASET_CONFIG['sampling_rate']

    def __len__(self) -> int:
        return 1000 if self.split == "train" else len(self.files)

    def preprocess_audio(self, audio: np.ndarray) -> torch.Tensor:
        """
        Preprocess audio data for model input.
        
        Args:
            audio: Raw audio data
            
        Returns:
            Processed audio tensor
        """
        if len(audio.shape) > 1:
            audio = audio.mean(axis=0)
        
        inputs = self.processor(
            audio, 
            sampling_rate=self.sampling_rate,
            return_tensors="pt"
        )
        return inputs.input_values.squeeze(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item
            
        Returns:
            Dictionary containing processed audio and labels
        """
        if self.split == "train":
            idx = random.randint(0, len(self.files) - 1)
        
        file = self.files[idx].strip()
        label = self.mapping[file.split("/")[0]]
        
        audio, _ = librosa.load(
            os.path.join(self.root, file), 
            sr=self.sampling_rate
        )
        audio = audio.astype("float32")
        
        if self.split == "train":
            if len(audio) < self.seg_length:
                audio = np.pad(audio, (0, self.seg_length - len(audio)))
            else:
                start = random.randint(0, len(audio) - self.seg_length - self.sampling_rate)
                audio = audio[start : start + self.seg_length]
        else:
            if len(audio) < self.seg_length:
                audio = np.pad(audio, (0, self.seg_length - len(audio)))
            else:
                start = (len(audio) - self.seg_length) // 2
                audio = audio[start : start + self.seg_length]
        
        audio_tensor = self.preprocess_audio(audio)
        
        return {
            'input_values': audio_tensor,
            'labels': torch.tensor(label, dtype=torch.long)
        }

class MERTGenreClassifier(nn.Module):
    """MERT-based neural network model for genre classification."""
    
    def __init__(self, model_name: str, embedding_size: int, num_genres: int):
        """
        Initialize the model.
        
        Args:
            model_name: Name of the pretrained MERT model
            embedding_size: Size of MERT embeddings
            num_genres: Number of genre classes
        """
        super().__init__()
        self.mert = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.classifier = nn.Linear(embedding_size, num_genres)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_values: torch.Tensor, 
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            input_values: Input audio tensors
            labels: Ground truth labels (optional)
            
        Returns:
            Dictionary containing loss and logits
        """
        outputs = self.mert(input_values)
        pooled_output = torch.mean(outputs.last_hidden_state, dim=1)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            
        return {'loss': loss, 'logits': logits} if loss is not None else logits

def create_model(model_version: str) -> tuple:
    """
    Create MERT model and processor based on version.
    
    Args:
        model_version: Version of MERT model ('95m' or '330m')
        
    Returns:
        Tuple containing (processor, model)
    """
    from config import MERT_CONFIGS
    
    config = MERT_CONFIGS[model_version]
    
    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        config['name'], 
        trust_remote_code=True
    )
    
    model = MERTGenreClassifier(
        model_name=config['name'],
        embedding_size=config['embedding_size'],
        num_genres=DATASET_CONFIG['num_classes']
    )
    
    return processor, model

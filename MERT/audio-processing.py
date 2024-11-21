#!/usr/bin/env python3
"""
Audio Dataset Processing Script for Genre Classification
This script processes audio files and prepares them for genre classification using the Wav2Vec2 model.
It handles loading audio files, preprocessing them, and creating PyTorch datasets.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio
from transformers import Wav2Vec2FeatureExtractor
from typing import Tuple, List, Dict, Optional
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AudioProcessor:
    """Handles audio file processing and dataset creation."""
    
    def __init__(self, data_dir: str, processor: Wav2Vec2FeatureExtractor):
        """
        Initialize the AudioProcessor.
        
        Args:
            data_dir (str): Base directory containing audio files
            processor (Wav2Vec2FeatureExtractor): Audio feature extractor
        """
        self.data_dir = data_dir
        self.processor = processor
        
    @staticmethod
    def load_paths_from_txt(txt_file: str) -> List[str]:
        """
        Load audio file paths from a text file.
        
        Args:
            txt_file (str): Path to text file containing audio file paths
            
        Returns:
            List[str]: List of file paths
        """
        try:
            with open(txt_file, 'r') as f:
                paths = [line.strip() for line in f.readlines()]
            return paths
        except FileNotFoundError:
            logger.error(f"File not found: {txt_file}")
            raise
        except Exception as e:
            logger.error(f"Error reading file {txt_file}: {str(e)}")
            raise

    def create_dataset_from_txt(self, txt_file: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Create a DataFrame from text file paths and generate genre mapping.
        
        Args:
            txt_file (str): Path to text file containing audio file paths
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, int]]: DataFrame with audio files info and genre mapping
        """
        paths = self.load_paths_from_txt(txt_file)
        audio_files = []
        genre_mapping = {}
        
        for path in paths:
            try:
                genre = path.split('/')[0]
                if genre not in genre_mapping:
                    genre_mapping[genre] = len(genre_mapping)
                
                full_path = os.path.join(self.data_dir, path)
                
                # Verify file exists
                if not os.path.exists(full_path):
                    logger.warning(f"File not found: {full_path}")
                    continue
                
                audio_files.append({
                    'file': full_path,
                    'genre': genre,
                    'label': genre_mapping[genre]
                })
            except Exception as e:
                logger.error(f"Error processing path {path}: {str(e)}")
                continue
        
        return pd.DataFrame(audio_files), genre_mapping

    def preprocess_audio(self, audio_file: str) -> Optional[torch.Tensor]:
        """
        Preprocess audio file for model input.
        
        Args:
            audio_file (str): Path to audio file
            
        Returns:
            Optional[torch.Tensor]: Processed audio tensor or None if processing fails
        """
        try:
            waveform, sample_rate = torchaudio.load(audio_file)
            
            # Resample if necessary
            if sample_rate != self.processor.sampling_rate:
                resampler = torchaudio.transforms.Resample(
                    sample_rate, 
                    self.processor.sampling_rate
                )
                waveform = resampler(waveform)
            
            # Convert to mono if necessary
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Process audio
            input_values = self.processor(
                waveform.squeeze().numpy(), 
                sampling_rate=self.processor.sampling_rate, 
                return_tensors="pt"
            ).input_values
            
            return input_values.squeeze()
        except Exception as e:
            logger.error(f"Error processing {audio_file}: {str(e)}")
            return None

class AudioDataset(Dataset):
    """PyTorch Dataset for audio files."""
    
    def __init__(self, dataframe: pd.DataFrame, audio_processor: AudioProcessor):
        """
        Initialize the AudioDataset.
        
        Args:
            dataframe (pd.DataFrame): DataFrame containing audio file information
            audio_processor (AudioProcessor): Processor for audio files
        """
        self.dataframe = dataframe
        self.audio_processor = audio_processor

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a dataset item.
        
        Args:
            idx (int): Index of the item
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing processed audio and labels
        """
        row = self.dataframe.iloc[idx]
        audio_tensor = self.audio_processor.preprocess_audio(row['file'])
        
        if audio_tensor is None:
            # Return a zero tensor of the expected shape if processing fails
            audio_tensor = torch.zeros((1, self.audio_processor.processor.sampling_rate))
        
        return {
            'input_values': audio_tensor,
            'label': torch.tensor(row['label']),
            'genre': row['genre']
        }

def main():
    """Main function to create and process datasets."""
    # Configuration
    data_dir = '/l/users/muhammad.airlangga/LMGC/dataset/'
    train_txt = os.path.join(data_dir, 'train_filtered.txt')
    val_txt = os.path.join(data_dir, 'valid_filtered.txt')
    test_txt = os.path.join(data_dir, 'test_filtered.txt')
    
    try:
        # Initialize processor and audio processor
        processor = Wav2Vec2FeatureExtractor.from_pretrained(
            "m-a-p/MERT-v1-95M", 
            trust_remote_code=True
        )
        audio_processor = AudioProcessor(data_dir, processor)
        
        # Load datasets
        logger.info("Loading datasets...")
        train_df, genre_mapping = audio_processor.create_dataset_from_txt(train_txt)
        val_df, _ = audio_processor.create_dataset_from_txt(val_txt)
        test_df, _ = audio_processor.create_dataset_from_txt(test_txt)
        
        # Create datasets
        train_dataset = AudioDataset(train_df, audio_processor)
        val_dataset = AudioDataset(val_df, audio_processor)
        test_dataset = AudioDataset(test_df, audio_processor)
        
        # Print dataset information
        logger.info("\nDataset splits:")
        logger.info(f"Train set size: {len(train_df)}")
        logger.info(f"Validation set size: {len(val_df)}")
        logger.info(f"Test set size: {len(test_df)}")
        logger.info("\nGenre mapping: %s", genre_mapping)
        
        # Print class distribution
        logger.info("\nClass distribution in splits:")
        logger.info("\nTrain set:\n%s", train_df['genre'].value_counts(normalize=True))
        logger.info("\nValidation set:\n%s", val_df['genre'].value_counts(normalize=True))
        logger.info("\nTest set:\n%s", test_df['genre'].value_counts(normalize=True))
        
        return train_dataset, val_dataset, test_dataset, genre_mapping
        
    except Exception as e:
        logger.error(f"An error occurred in main: {str(e)}")
        raise

if __name__ == "__main__":
    train_dataset, val_dataset, test_dataset, genre_mapping = main()


import os
import numpy as np
from torch.utils import data
import random
import soundfile as sf
import librosa
import librosa.effects as effects


class GTZAN(data.Dataset):
    def __init__(self, root, split, input_length=None, augment=True):
        split = split.lower()
        self.mappeing = {
            "Campursari": 0,
            "Dangdut": 1,
            "Keroncong": 2,
            "Baila": 3,
            "Wannam": 4,
            "Virindu": 5,
            "Qawwali": 6,
            "Pop": 7,
            "Ghazal": 8,
        }
        self.files = [
            f
            for f in open(f"{root}/{split}_filtered.txt", "r").readlines()
            if "jazz.00054" not in f
        ]
        self.class_num = 9
        self.split = split
        self.seg_length = input_length
        self.root = root
        self.augment = augment

    def __len__(self):
        if self.split == "train":
            return 1000
        else:
            return len(self.files)

    def augment_audio(self, audio):

        
        #Pitch shifting
        # if random.random() < 0.5:
        #     pitch_shift_steps = random.randint(-1, 1)
        #     audio = effects.pitch_shift(audio, sr=16000, n_steps=pitch_shift_steps)
        
     #   Additive noise
        if random.random() < 0.5:
            # noise = np.random.normal(0,0.3, audio.shape)
            # audio += noise
            audio += 0.5*audio
        # Random volume adjustment
        if random.random() < 0.5:
            volume_factor = random.uniform(0.7, 1.3)
            audio *= volume_factor
        
        return audio

    def __getitem__(self, idx):
        if self.split == "train":
            idx = random.randint(0, len(self.files) - 1)
        file = self.files[idx].strip()
        frame = sf.info(os.path.join(self.root, file)).frames
        label = np.zeros(self.class_num)
        label[self.mappeing[file.split("/")[0]]] = 1

        # Load audio file
        audio, sr = librosa.load(os.path.join(self.root, file), sr=16000)

        # Apply augmentations if training and augment flag is True
        if self.split == "train" and self.augment:
            audio = self.augment_audio(audio)

        # Random segment selection for training
        if self.split == "train":
            start = random.randint(0, len(audio) - self.seg_length - 16000)
            audio = audio[start : start + self.seg_length]
            audio = audio.astype("float32")
            return audio, label.astype("float32")
        
        # For validation/testing, chunk the audio
        else:
            audio = audio.astype("float32")
            n_chunk = len(audio) // self.seg_length
            audio_chunks = np.split(audio[: int(n_chunk * self.seg_length)], n_chunk)
            audio_chunks.append(audio[-int(self.seg_length) :])
            audio = np.array(audio_chunks)

            return audio, label.astype("float32")


def get_audio_loader(
    root,
    batch_size,
    split="TRAIN",
    num_workers=0,
    input_length=None,
    augment=True
):
    data_loader = data.DataLoader(
        dataset=GTZAN(root, split=split, input_length=input_length, augment=augment),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    return data_loader

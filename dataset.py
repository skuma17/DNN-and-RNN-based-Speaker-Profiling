import torch
import torchaudio
from torch.utils.data import Dataset
import wavencoder
import numpy as np
import os
import pandas as pd
import speechbrain
from speechbrain.lobes.models.Xvector import Xvector



class NISPDataset(Dataset):
    def __init__(self,
        wav_folder,
        csv_file,
        wav_len=48000,
        is_train=True,
        noise_dataset_path=None
        ):
        
        self.wav_folder = wav_folder
        self.files = os.listdir(self.wav_folder)
        self.csv_file = csv_file
        self.df = pd.read_csv(self.csv_file, sep=' ')
        self.is_train = is_train
        self.wav_len = wav_len
        self.noise_dataset_path = noise_dataset_path

        self.gender_dict = {'Male':0,
                            'Female':1}
        self.nl_dict = {'Hindi':0,
                        'Tamil':1,
                        'Telugu':2,
                        'malayalam':3,
                        'Kannada':4 }
        

        if self.noise_dataset_path:

           self.train_transform = wavencoder.transforms.Compose([
                wavencoder.transforms.PadCrop(pad_crop_length=self.wav_len, pad_position='left', crop_position='random'),
                wavencoder.transforms.AdditiveNoise(self.noise_dataset_path, p=0.5),
                wavencoder.transforms.Clipping(p=0.5),
                wavencoder.transforms.SpeedChange(factor_range=(-0.5, 0.0), p=0.5),                
                ])
        else:
            self.train_transform = wavencoder.transforms.Compose([
                  wavencoder.transforms.PadCrop(pad_crop_length=self.wav_len, pad_position='left', crop_position='random'),
                  wavencoder.transforms.Clipping(p=0.5),
                  ])

            self.test_transform = wavencoder.transforms.Compose([
              wavencoder.transforms.PadCrop(pad_crop_length=self.wav_len)
              ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        file = self.files[idx]
        id = file[:8]
        
        wav, sample_rate = torchaudio.load(os.path.join(self.wav_folder, file))

        if sample_rate != 16000:
            wav = torchaudio.functional.resample(wav, sample_rate, 16000)
        
        
        if self.is_train:
            wav = self.train_transform(wav)
        else:
            wav = self.test_transform(wav)
          

        df_row = self.df[self.df['Speaker_ID'] == id]
        gender = self.gender_dict[str(df_row['Gender'].values.item())]
        native_languages = self.nl_dict[df_row['Mother_Tongue'].values.item()]
        height = df_row['Height'].values.item()
        shoulder_size = df_row['Shoulder_size'].values.item()
        waist_size = df_row['Waist_size'].values.item()
        weight = df_row['Weight'].values.item()
        age = df_row['Age'].values.item()

        height = (height - self.df['Height'].mean())/self.df['Height'].std()
        age = (age - self.df['Age'].mean())/self.df['Age'].std()
        return wav, native_languages, age, gender
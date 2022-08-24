import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy

import wavencoder

import speechbrain
from speechbrain.lobes.features import MFCC, Fbank
from speechbrain.lobes.models.Xvector import Xvector
from speechbrain.lobes.models.Xvector import Classifier
from speechbrain.lobes.models.Xvector import Discriminator

class SpeechBrainLSTM(nn.Module):
    def __init__(self, lstm_h, lstm_inp=512):
        super().__init__()

        self.feature_maker = Fbank()
        self.compute_xvect = Xvector(device='gpu') 

        self.lstm = nn.LSTM(lstm_inp, lstm_h, batch_first=True)
 
        self.native_languages_classifier = nn.Sequential(
            nn.Linear(lstm_h, lstm_h),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(lstm_h, 5),
            nn.LogSoftmax(dim=1)
        )           
        self.age_regressor = nn.Sequential(
            nn.Linear(lstm_h, lstm_h),
            nn.ReLU(),
            nn.Linear(lstm_h, 1))


        self.gender_classifier = nn.Sequential(
            nn.Linear(lstm_h, lstm_h),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(lstm_h, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        x = x.squeeze(1)
        feats = self.feature_maker(x)        
        xvector = self.compute_xvect(feats.float())         
        output, (hidden, _) = self.lstm(xvector)
        attn_output = output.squeeze(1)              
        native_languages = self.native_languages_classifier(attn_output)
        age = self.age_regressor(attn_output)
        gender = self.gender_classifier(attn_output)
        return native_languages, age, gender

  

class Wav2VecLSTM_Base(nn.Module):
    def __init__(self, lstm_h, lstm_inp=512):
        super().__init__()
        self.encoder = wavencoder.models.Wav2Vec(pretrained=True)

        for param in self.encoder.parameters():
            param.requires_grad = False

        for param in self.encoder.feature_extractor.conv_layers[5:].parameters():
            param.requires_grad = True

               
        self.lstm = nn.LSTM(lstm_inp, lstm_h, batch_first=True)
        self.attention = wavencoder.layers.SoftAttention(lstm_h, lstm_h) 
 
        self.native_languages_classifier = nn.Sequential(
            nn.Linear(lstm_h, lstm_h),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(lstm_h, 5),
            nn.LogSoftmax(dim=1)
        )           
        self.age_regressor = nn.Linear(lstm_h, 1)
        self.gender_classifier = nn.Sequential(
            nn.Linear(lstm_h, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)

        output, (hidden, _) = self.lstm(x.transpose(1,2))
        attn_output = self.attention(output)
        
        native_languages = self.native_languages_classifier(attn_output)
        age = self.age_regressor(attn_output)
        gender = self.gender_classifier(attn_output)
        return native_languages, age, gender
     
    

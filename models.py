import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy

import wavencoder

import speechbrain
from speechbrain.lobes.models.Xvector import Xvector
from speechbrain.lobes.models.Xvector import Classifier
from speechbrain.lobes.models.Xvector import Discriminator


class Wav2VecXectorLSTM(nn.Module):
    def __init__(self, lstm_h, lstm_inp=512):
        super().__init__()

        self.lstm = nn.LSTM(lstm_inp, lstm_h, batch_first=True)
        for param in self.lstm.parameters():
            param.requires_grad=True
    
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
        x = x.squeeze(2)
        print ("shape of x", x.shape)

        output, (hidden, _) = self.lstm(x.transpose(1,2))
        attn_output = self.attention(output)
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
# Size of the mfcc tensor[128, 512, 498]
        output, (hidden, _) = self.lstm(x.transpose(1,2))
        attn_output = self.attention(output)
        
        native_languages = self.native_languages_classifier(attn_output)
        age = self.age_regressor(attn_output)
        gender = self.gender_classifier(attn_output)
        return native_languages, age, gender
     
    
    
class Wav2VecLSTM(nn.Module):
    def __init__(self, lstm_h, lstm_inp=512):
        super().__init__()
        self.encoder = wavencoder.models.Wav2Vec(pretrained=True)
        for param in self.encoder.parameters():
            param.requires_grad = False

        for param in self.encoder.feature_extractor.conv_layers[5:].parameters():
            param.requires_grad = True
        
        self.lstm = nn.LSTM(lstm_inp, lstm_h, batch_first=True)
        self.attention = wavencoder.layers.SoftAttention(lstm_h, lstm_h)
        self.height_regressor = nn.Linear(lstm_h, 1)
        self.age_regressor = nn.Linear(lstm_h, 1)
        self.gender_classifier = nn.Sequential(
            nn.Linear(lstm_h, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        output, (hidden, _) = self.lstm(x.transpose(1,2))
        attn_output = self.attention(output)
        height = self.height_regressor(attn_output)
        age = self.age_regressor(attn_output)
        gender = self.gender_classifier(attn_output)
        return height, age, gender

class SpectralLSTM(nn.Module):
    def __init__(self, lstm_h):
        super().__init__()

        self.lstm = nn.LSTM(128, lstm_h, batch_first=True)
        self.attention = wavencoder.layers.SoftAttention(lstm_h, lstm_h)
    
        self.height_regressor = nn.Sequential(
            nn.Linear(lstm_h, lstm_h),
            nn.ReLU(),
            nn.Linear(lstm_h, 1))
        self.age_regressor = nn.Sequential(
            nn.Linear(lstm_h, lstm_h),
            nn.ReLU(),
            nn.Linear(lstm_h, 1))
        self.gender_classifier = nn.Sequential(
            nn.Linear(lstm_h, lstm_h),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(lstm_h, 1),
            nn.Sigmoid())

    def forward(self, x):
        x = x.squeeze(1)
        output, (hidden, _) = self.lstm(x.transpose(1,2))
        attn_output = self.attention(output)

        height = self.height_regressor(attn_output)
        age = self.age_regressor(attn_output)
        gender = self.gender_classifier(attn_output)
        return height, age, gender

class SpectralCNNLSTM(nn.Module):
    def __init__(self, lstm_h):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(40, int(lstm_h/2), 5),
            nn.ReLU(),
            nn.BatchNorm1d(int(lstm_h/2)),
            nn.Conv1d(int(lstm_h/2), int(lstm_h/2), 5),
            nn.ReLU(),
            nn.BatchNorm1d(int(lstm_h/2)),
        )
        self.lstm = nn.LSTM(int(lstm_h/2), int(lstm_h/2), batch_first=True)
        self.attention = wavencoder.layers.SoftAttention(int(lstm_h/2), int(lstm_h/2))
    
        self.height_regressor = nn.Sequential(
            nn.Linear(int(lstm_h/2), int(lstm_h/2)),
            nn.ReLU(),
            nn.Linear(int(lstm_h/2), 1))
        self.age_regressor = nn.Sequential(
            nn.Linear(int(lstm_h/2), int(lstm_h/2)),
            nn.ReLU(),
            nn.Linear(int(lstm_h/2), 1))
        self.gender_classifier = nn.Sequential(
            nn.Linear(int(lstm_h/2), int(lstm_h/2)),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(int(lstm_h/2), 1),
            nn.Sigmoid())

    def forward(self, x):
        x = x.squeeze(1)
        x = self.encoder(x)
        output, (hidden, _) = self.lstm(x.transpose(1,2))
        attn_output = self.attention(output)

        height = self.height_regressor(attn_output)
        age = self.age_regressor(attn_output)
        gender = self.gender_classifier(attn_output)
        return height, age, gender

class SpectralMultiScale(nn.Module):
    def __init__(self, lstm_h):
        super().__init__()
        inp_dim = 40
        self.lstm_h = lstm_h
        self.cnn3 = nn.Sequential(
            nn.BatchNorm1d(inp_dim),
            nn.Conv1d(inp_dim, lstm_h, 3),
            nn.ReLU(),
            nn.BatchNorm1d(lstm_h),
            nn.MaxPool1d(2,2),
            nn.Conv1d(lstm_h, lstm_h, 3),
            nn.ReLU(),
            nn.BatchNorm1d(lstm_h),
            nn.MaxPool1d(2,2),
            nn.Conv1d(lstm_h, lstm_h, 3),
            nn.ReLU(),
            nn.BatchNorm1d(lstm_h),
            nn.MaxPool1d(2,2),
            TransposeAttn(lstm_h)  
        )

        self.cnn5 = nn.Sequential(
            nn.BatchNorm1d(inp_dim),
            nn.Conv1d(inp_dim, lstm_h, 5),
            nn.ReLU(),
            nn.BatchNorm1d(lstm_h),
            nn.MaxPool1d(2,2),
            nn.Conv1d(lstm_h, lstm_h, 5),
            nn.ReLU(),
            nn.BatchNorm1d(lstm_h),
            nn.MaxPool1d(2,2),
            nn.Conv1d(lstm_h, lstm_h, 5),
            nn.ReLU(),
            nn.BatchNorm1d(lstm_h),
            nn.MaxPool1d(2,2),
            TransposeAttn(lstm_h)
        )

        self.cnn7 = nn.Sequential(
            nn.BatchNorm1d(inp_dim),
            nn.Conv1d(inp_dim, lstm_h, 7),
            nn.ReLU(),
            nn.BatchNorm1d(lstm_h),
            nn.MaxPool1d(2,2),
            nn.Conv1d(lstm_h, lstm_h, 7),
            nn.ReLU(),
            nn.BatchNorm1d(lstm_h),
            nn.MaxPool1d(2,2),
            nn.Conv1d(lstm_h, lstm_h, 7),
            nn.ReLU(),
            nn.BatchNorm1d(lstm_h),
            nn.MaxPool1d(2,2),
            TransposeAttn(lstm_h)
        )
        self.height_regressor = nn.Sequential(
            nn.Linear(3*lstm_h, lstm_h),
            nn.ReLU(),
            nn.Linear(lstm_h, 1),
        )

        self.age_regressor = nn.Sequential(
            nn.Linear(3*lstm_h, lstm_h),
            nn.ReLU(),
            nn.Linear(lstm_h, 1),
        )

        self.gender_regressor = nn.Sequential(
            nn.Linear(3*lstm_h, lstm_h),
            nn.ReLU(),
            nn.Linear(lstm_h, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(1)

        fm3 = self.cnn3(x).view(-1, self.lstm_h)
        fm5 = self.cnn5(x).view(-1, self.lstm_h)
        fm7 = self.cnn7(x).view(-1, self.lstm_h)

        fm = torch.cat([fm3, fm5, fm7], 1)
        height = self.height_regressor(fm)
        age = self.age_regressor(fm)
        gender = self.gender_regressor(fm)
        return height, age, gender


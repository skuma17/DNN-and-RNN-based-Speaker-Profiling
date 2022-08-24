import torch
import torch.nn as nn
import torch.nn.functional as F

import torchaudio

import wavencoder
import torchmetrics
from pytorch_lightning import LightningModule
import pytorch_lightning as pl

from torchmetrics import Accuracy, ConfusionMatrix
from torchmetrics import F1Score, AUROC
from torchmetrics import MeanSquaredError  as MSE
from torchmetrics import MeanAbsoluteError as MAE



from models import Wav2VecLSTM_Base
from models import SpeechBrainLSTM


import pandas as pd
import wavencoder

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))

class LightningModel(pl.LightningModule):
    def __init__(self, HPARAMS):
        super(LightningModel, self).__init__()
        # HPARAMS
        self.save_hyperparameters()
        self.model = SpeechBrainLSTM(HPARAMS['model_hidden_size'])
        #self.model = Wav2VecLSTM_Base(HPARAMS['model_hidden_size'])

        self.ConfutionMatrix_MultiClass_criterion = ConfusionMatrix(num_classes=5)
        self.ConfutionMatrix_BinaryClass_criterion = ConfusionMatrix(num_classes=2, threshold=0.5, Multilabel=False )
        self.F1_criterion = F1Score(number_classes=5,
        average="micro")
        self.AUC_criterion = AUROC(num_classes=5,
        average="micro")

        self.nl_numclass = 5

        self.classification_criterion = MSE()
        self.regression_criterion = MSE()
        self.mae_criterion = MAE()
        self.rmse_criterion = RMSELoss()
        self.accuracy = Accuracy()

        self.alpha = HPARAMS['model_alpha']
        self.beta = HPARAMS['model_beta']
        self.gamma = HPARAMS['model_gamma']

        self.lr = HPARAMS['training_lr']

        
        self.csv_path = HPARAMS['speaker_csv_path']
        self.df = pd.read_csv(self.csv_path, sep=' ')
        
        self.a_mean = self.df['Age'].mean()
        self.a_std = self.df['Age'].std()
        
        print(f"Model Details: #Params = {self.count_total_parameters()}\t#Trainable Params = {self.count_trainable_parameters()}")

    def count_total_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
  
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return [optimizer]

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, batch, batch_idx):
        x, y_nl, y_a, y_g = batch
        y_hat_nl, y_hat_a, y_hat_g = self(x)

        y_a, y_g = y_a.view(-1).float(), y_g.view(-1).float()
        y_hat_a, y_hat_g = y_hat_a.view(-1).float(), y_hat_g.view(-1).float()

        native_languages_loss = self.cross_entropy_loss(y_hat_nl, y_nl)            
        age_loss = self.regression_criterion(y_hat_a, y_a)
        gender_loss = self.classification_criterion(y_hat_g, y_g)
        loss = self.alpha * native_languages_loss + self.beta * age_loss + self.gamma * gender_loss

        native_languages_acc = self.accuracy(y_hat_nl.float(), y_nl)
        age_mae =self.mae_criterion(y_hat_a*self.a_std+self.a_mean, y_a*self.a_std+self.a_mean)
        age_rmse =self.rmse_criterion(y_hat_a*self.a_std+self.a_mean, y_a*self.a_std+self.a_mean)
        gender_acc = self.accuracy((y_hat_g>0.5).long(), y_g.long())
        nl_F1Score = self.F1_criterion(y_hat_nl.float(), y_nl)
        g_F1Score = self.F1_criterion((y_hat_g>0.5).long(), y_g.long())
        
        #y_hat_nl = y_hat_nl.argmax(axis=1)
        #nl_confmatrix = self.ConfutionMatrix_MultiClass_criterion.update(y_hat_nl, y_nl )
        #g_confmatrix = self.ConfutionMatrix_BinaryClass_criterion.update(y_hat_g, y_g)
        #nl_auroc = self.AUC_criterion(y_hat_nl.float(), y_nl)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)

        return {'loss':loss, 
                'train_native_languages_acc':native_languages_acc,
                'train_age_mae':age_mae.item(),
                'train_age_rmse':age_rmse.item(),
                'train_gender_acc':gender_acc,
                'train_nl_F1score':nl_F1Score,
                'train_g_F1score':g_F1Score,
#                'train_nl_Confmatrix':nl_confmatrix,
#                'train_g_Confmatrix':g_confmatrix,
#                'train_nl_auroc':nl_auroc,
                 }
    
    def training_epoch_end(self, outputs):
        n_batch = len(outputs)
        loss = torch.tensor([x['loss'] for x in outputs]).mean()
                
        native_languages_acc = torch.tensor([x['train_native_languages_acc'] for x in outputs]).mean()
        age_mae = torch.tensor([x['train_age_mae'] for x in outputs]).sum()/n_batch
        age_rmse = torch.tensor([x['train_age_rmse'] for x in outputs]).sum()/n_batch
        gender_acc = torch.tensor([x['train_gender_acc'] for x in outputs]).mean()
        nl_F1Score = torch.tensor([x['train_nl_F1score'] for x in outputs]).mean()
        g_F1Score = torch.tensor([x['train_g_F1score'] for x in outputs]).mean()
        #nl_confmatrix = torch.tensor([x['train_nl_Confmatrix'] for x in outputs]).mean()
        #nl_confmatrix = self.ConfutionMatrix_MultiClass_criterion.compute()
        #g_confmatrix = self.ConfutionMatrix_BinaryClass_criterion.compute()
        #nl_auroc = torch.tensor([x['train_nl_auroc'] for x in outputs]).mean()

        self.log('epoch_loss' , loss, prog_bar=True, sync_dist=True)
        self.log('native-language accuracy',native_languages_acc, prog_bar=True, sync_dist=True)
        self.log('Age_mae',age_mae.item(), prog_bar=True, sync_dist=True)
        self.log('Age_rmse',age_rmse.item(), prog_bar=True, sync_dist=True)
        self.log('gender_accuracy',gender_acc, prog_bar=True, sync_dist=True)
        self.log('native-language F1Score',nl_F1Score, prog_bar=True, sync_dist=True)
        self.log('gender F1Score',g_F1Score, prog_bar=True, sync_dist=True)
        #self.log('g_Confmatrix',g_confmatrix, prog_bar=True, sync_dist=True)
        #self.log('nl_auroc',nl_auroc, prog_bar=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        x, y_nl, y_a, y_g = batch       
        y_hat_nl, y_hat_a, y_hat_g = self(x)

        y_a, y_g = y_a.view(-1).float(), y_g.view(-1).float()
        y_hat_a, y_hat_g = y_hat_a.view(-1).float(), y_hat_g.view(-1).float()

        native_languages_loss =  self.cross_entropy_loss(y_hat_nl, y_nl)        
        age_loss = self.regression_criterion(y_hat_a, y_a)
        gender_loss = self.classification_criterion(y_hat_g, y_g)
        loss = self.alpha * native_languages_loss + self.beta * age_loss + self.gamma * gender_loss
        
        native_languages_acc = self.accuracy(y_hat_nl.float(), y_nl)        
        age_mae =self.mae_criterion(y_hat_a*self.a_std+self.a_mean, y_a*self.a_std+self.a_mean)
        age_rmse =self.rmse_criterion(y_hat_a*self.a_std+self.a_mean, y_a*self.a_std+self.a_mean)
        gender_acc = self.accuracy((y_hat_g>0.5).long(), y_g.long())

        return {'val_loss':loss, 
                'val_native_languages_acc':native_languages_acc,
                'val_age_mae':age_mae.item(),
                'val_age_rmse':age_rmse.item(),
                'val_gender_acc':gender_acc}


    def validation_epoch_end(self, outputs):
        n_batch = len(outputs)
        val_loss = torch.tensor([x['val_loss'] for x in outputs]).mean()
        
        native_languages_acc = torch.tensor([x['val_native_languages_acc'] for x in outputs]).mean()
        age_mae = torch.tensor([x['val_age_mae'] for x in outputs]).sum()/n_batch
        age_rmse = torch.tensor([x['val_age_rmse'] for x in outputs]).sum()/n_batch
        gender_acc = torch.tensor([x['val_gender_acc'] for x in outputs]).mean()

        self.log('v_loss' , val_loss, prog_bar=True, sync_dist=True)
        self.log('v_nl_acc',native_languages_acc, prog_bar=True, sync_dist=True)
        self.log('v_a_mae',age_mae.item(), prog_bar=True, sync_dist=True)
        self.log('v_a_rmse',age_rmse.item(), prog_bar=True, sync_dist=True)
        self.log('v_g_acc',gender_acc, prog_bar=True, sync_dist=True)
        
        
    def test_step(self, batch, batch_idx):       
        x, y_nl, y_a, y_g = batch
        y_hat_nl, y_hat_a, y_hat_g = self(x)

        y_a, y_g = y_a.view(-1).float(), y_g.view(-1).float()
        y_hat_a, y_hat_g = y_hat_a.view(-1).float(), y_hat_g.view(-1).float()

        native_languages_acc = self.accuracy(y_hat_nl.float(), y_nl)  
        gender_acc = self.accuracy((y_hat_g>0.5).long(), y_g.long())

        idx = y_g.view(-1).long()
        female_idx = torch.nonzero(idx).view(-1)
        male_idx = torch.nonzero(1-idx).view(-1)
        
        male_age_mae = self.mae_criterion(y_hat_a[male_idx]*self.a_std+self.a_mean, y_a[male_idx]*self.a_std+self.a_mean)      
        male_age_mae = torch.nan_to_num(male_age_mae)
        female_age_mae = self.mae_criterion(y_hat_a[female_idx]*self.a_std+self.a_mean, y_a[female_idx]*self.a_std+self.a_mean)
        female_age_mae = torch.nan_to_num(female_age_mae)
        
        male_age_rmse = self.rmse_criterion(y_hat_a[male_idx]*self.a_std+self.a_mean, y_a[male_idx]*self.a_std+self.a_mean)
        male_age_rmse = torch.nan_to_num(male_age_rmse)
        female_age_rmse = self.rmse_criterion(y_hat_a[female_idx]*self.a_std+self.a_mean, y_a[female_idx]*self.a_std+self.a_mean)
        female_age_rmse = torch.nan_to_num(female_age_rmse)
        
        return {
                'test_native_languages_acc':native_languages_acc,
                'male_age_mae':male_age_mae.item(),
                'female_age_mae':female_age_mae.item(),
                'male_age_rmse':male_age_rmse.item(),
                'female_age_rmse':female_age_rmse.item(),
                'test_gender_acc':gender_acc}


    def test_epoch_end(self, outputs):
        n_batch = len(outputs)

        male_age_mae = torch.tensor([x['male_age_mae'] for x in outputs]).mean()
        female_age_mae = torch.tensor([x['female_age_mae'] for x in outputs]).mean()

        male_age_rmse = torch.tensor([x['male_age_rmse'] for x in outputs]).mean()
        female_age_rmse = torch.tensor([x['female_age_rmse'] for x in outputs]).mean()
        
        native_languages_acc = torch.tensor([x['test_native_languages_acc'] for x in outputs]).mean()        
        gender_acc = torch.tensor([x['test_gender_acc'] for x in outputs]).mean()

        pbar = {'test_native_languages_acc':native_languages_acc.item(),
                'male_age_mae':male_age_mae.item(),
                'female_age_mae': female_age_mae.item(),
                'male_age_rmse':male_age_rmse.item(),                
                'female_age_rmse': female_age_rmse.item(),
                'test_gender_acc':gender_acc.item()} 

        self.logger.log_hyperparams(pbar)
        self.log_dict(pbar)
        


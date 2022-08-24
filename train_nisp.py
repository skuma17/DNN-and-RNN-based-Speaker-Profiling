from argparse import ArgumentParser
from multiprocessing import Pool
import os

from dataset import NISPDataset
from lightning_model import LightningModel 

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
import wandb


from config import NISPConfig
import torch
import torch.utils.data as data

if __name__ == "__main__":

    parser = ArgumentParser(add_help=True)
    parser.add_argument('--data_path', type=str, default=NISPConfig.data_path)
    parser.add_argument('--speaker_csv_path', type=str, default=NISPConfig.speaker_csv_path)
    parser.add_argument('--nisp_wav_len', type=int, default=NISPConfig.nisp_wav_len)
    parser.add_argument('--batch_size', type=int, default=NISPConfig.batch_size)
    parser.add_argument('--num_languages', default=NISPConfig.num_languages)
    parser.add_argument('--input_dim', default=NISPConfig.input_dim)
    parser.add_argument('--epochs', type=int, default=NISPConfig.epochs)
    parser.add_argument('--alpha', type=float, default=NISPConfig.alpha)
    parser.add_argument('--beta', type=float, default=NISPConfig.beta)
    parser.add_argument('--gamma', type=float, default=NISPConfig.gamma)
    parser.add_argument('--hidden_size', type=float, default=NISPConfig.hidden_size)
    parser.add_argument('--gpu', type=int, default=NISPConfig.gpu)
    parser.add_argument('--n_workers', type=int, default=NISPConfig.n_workers)
    parser.add_argument('--dev', type=str, default=False)
    parser.add_argument('--model_checkpoint', type=str, default=NISPConfig.model_checkpoint)
    parser.add_argument('--model_type', type=str, default=NISPConfig.model_type)
    parser.add_argument('--data_type', type=str, default=NISPConfig.data_type)
    parser.add_argument('--noise_dataset_path', type=str, default=NISPConfig.noise_dataset_path)

    parser = pl.Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    print(f'Training Model on NISP Dataset\n#Cores = {hparams.n_workers}\t#GPU = {hparams.gpu}')

    # hyperparameters and details about the model 
    HPARAMS = {
        'data_path' : hparams.data_path,
        'speaker_csv_path' : hparams.speaker_csv_path,
        'data_wav_len' : hparams.nisp_wav_len,
        'data_batch_size' : hparams.batch_size,
        'data_wav_augmentation' : 'Random Crop, Additive Noise',
        'data_label_scale' : 'Standardization',

        'training_optimizer' : 'Adam',
        'training_lr' : NISPConfig.lr,
        'training_lr_scheduler' : '-',

        'model_hidden_size' : hparams.hidden_size,
        'model_alpha' : hparams.alpha,
        'model_beta' : hparams.beta,
        'model_gamma' : hparams.gamma,
        'model_architecture' : 'wav2vec + soft-attention',
    }

    # Training, Validation and Testing Dataset
    ## Training Dataset
    train_set = NISPDataset(
        wav_folder = os.path.join(HPARAMS['data_path'], 'TRAIN'),
        csv_file = HPARAMS['speaker_csv_path'],
        wav_len = HPARAMS['data_wav_len'],
        noise_dataset_path = hparams.noise_dataset_path
    )
    ## Training DataLoader
    trainloader = data.DataLoader(
        train_set, 
        batch_size=HPARAMS['data_batch_size'], 
        shuffle=True, 
        num_workers=hparams.n_workers
    )
    ## Validation Dataset
    valid_set = NISPDataset(
        wav_folder = os.path.join(HPARAMS['data_path'], 'VAL'),
        csv_file = HPARAMS['speaker_csv_path'],
        wav_len = HPARAMS['data_wav_len'],
        is_train=False
    )
    ## Validation Dataloader
    valloader = data.DataLoader(
        valid_set, 
        batch_size=HPARAMS['data_batch_size'], 
        shuffle=False, 
        num_workers=hparams.n_workers
    )
    ## Testing Dataset
    test_set = NISPDataset(
        wav_folder = os.path.join(HPARAMS['data_path'], 'TEST'),
        csv_file = HPARAMS['speaker_csv_path'],
        wav_len = HPARAMS['data_wav_len'],
        is_train=False
    )
    ## Testing Dataloader
    testloader = data.DataLoader(
        test_set, 
        batch_size=HPARAMS['data_batch_size'], 
        shuffle=False, 
        num_workers=hparams.n_workers
    )

    print('Dataset Split (Train, Validation, Test)=', len(train_set), len(valid_set), len(test_set))

  #  device = "cuda" if torch.cuda.is_available() else "cpu"
  #  print(f"Using {device} device")

    #Training the Model
     
    wandb_logger = WandbLogger(project="SpeakerProfiling",log_model="all")   

    model = LightningModel(HPARAMS)

    checkpoint_callback = ModelCheckpoint(
        monitor='v_loss', 
        mode='min',
        verbose=1)
    early_stop= EarlyStopping(
                        monitor='v_loss',
                        min_delta=0.00,
                        patience=20,
                        verbose=True,
                        mode='min' )

    trainer = pl.Trainer(fast_dev_run=hparams.dev, 
                        max_epochs=hparams.epochs, 
                        callbacks=[checkpoint_callback, early_stop ],
                        logger=wandb_logger,
                        resume_from_checkpoint=hparams.model_checkpoint,
                        strategy='ddp',
                        accelerator="gpu",
                        devices=1,
                        )

    # log gradients and model topology
    wandb_logger.watch(model, log="all")

    trainer.fit(model, trainloader, valloader)
    
    wandb_logger.watch(model, log_graph=False)
   
    print("\n Completed Training...")
     
    hparams.model_checkpoint = checkpoint_callback.best_model_path
    
    print('Testing the model with checkpoint \n ',checkpoint_callback.best_model_path)
    if hparams.model_checkpoint!=None:
        model = LightningModel.load_from_checkpoint(checkpoint_callback.best_model_path)
        print('\nTesting on NISP Dataset:\n')
        trainer.test(model, testloader)
    else:
        print('Model check point for testing is not provided!!!')
    
import os


class NISPConfig(object):
    # path to the unzuipped NISP data folder
    
    data_path = "..\\NISP-Dataset\\final_data_16k"

    # path to csv file containing age, heights of NISP speakers
    speaker_csv_path = "..\\NISP-Dataset\\total_spkrinfo.list"

    # length of wav files for training and testing
    nisp_wav_len = 16000 * 5
    
    #input dimension
    input_dim = 257

    batch_size = 128
    epochs = 50
    
    # loss = alpha * height_loss + beta * age_loss + gamma * gender_loss
    alpha = 1
    beta = 1
    gamma = 1

    # data type - Wav2Vec
    data_type = 'Wav2Vec' 

    # model type
    
    model_type = 'Wav2VecLSTM_Base'
   

    ## number of language class
    num_languages = 5

    # hidden dimension of LSTM and Dense Layers
    hidden_size = 128

    # No of GPUs for training and no of workers for datalaoders
    gpu = '1'
    n_workers = 4

    # model checkpoint to continue from
    model_checkpoint = None

    # noise dataset for augmentation
    noise_dataset_path =  "..\\noise_dataset"

    # LR of optimizer
    lr = 1e-3
    
    run_name = data_type + '_' + model_type
    
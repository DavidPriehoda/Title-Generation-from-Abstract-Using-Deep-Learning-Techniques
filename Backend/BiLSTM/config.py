config = {
    'model_name': 'BiLSTM_model_6',
    'dataset_name': 'dataset.csv',
    'batch_size': 64,
    'num_layers': 2,
    'hidden_size': 128,
    'initial_epoch': 0, #Set this to the last epoch of the previous training, if you want to continue training. Set to 0 if you want to train from scratch.
    'epochs': 200,
    'max_len': 256, #Changing this will require re-running the preprocessing
    'vocab_size_titles': 8000, #Changing this will require re-running the preprocessing
    'vocab_size_abstracts': 6500, #Changing this will require re-running the preprocessing
    'validation_split': 0.2,
    'dropout': 0.5,
    'recurrent_dropout': 0.5,
    'dataset_dir': '../dataset/',
    'preprocessed_save_dir': './preprocessed/',
    'model_save_dir': './models/'
}

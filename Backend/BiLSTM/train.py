import wandb
from wandb.keras import WandbCallback

from utils import load_vocab_data, load_training_batch, create_model_folder
from config import config
from model import BiLSTMAttention

if __name__ == '__main__':
    
    ##Load parameters from config file
    titles_file = config['preprocessed_save_dir']+ config['model_name'] + '/titles.txt'
    abstracts_file = config['preprocessed_save_dir']+ config['model_name'] + '/abstracts.txt'
    vocab_file = config['preprocessed_save_dir']+ config['model_name'] + '/vocab.pkl'
    batch_size = config['batch_size']
    num_layers = config['num_layers']
    hidden_size = config['hidden_size']
    dropout = config['dropout']
    recurrent_dropout = config['recurrent_dropout']
    initial_epoch = config['initial_epoch']
    epochs = config['epochs']
    validation_split = config['validation_split']

    #Create model folder
    create_model_folder(config['model_save_dir'], config['model_name'])

    #Load vocab data
    word2idx_abstract, word2idx_title, idx2word_abstract, idx2word_title, max_len, vocab_size_title, vocab_size_abstract = load_vocab_data(vocab_file)

    #Create model
    model = BiLSTMAttention(vocab_size_abstract,max_len,vocab_size_title,max_len,hidden_size,num_layers, dropout, recurrent_dropout)

    #Load model if initial epoch is not 0
    if initial_epoch != 0:
        model_path = config['model_save_dir'] + '/' + config['model_name'] + '/' + 'epoch_{}'.format(initial_epoch)
        model.load(model_path)
    else:
        model.create_model()


    wandb.init(project="bilstm_attention", config=config)
    wandb_callback = WandbCallback(log_weights=True, save_model=False)

    # main training loop
    for e in range(initial_epoch+1, epochs+1):
        processed = 0
        for X_train, y_train in load_training_batch(titles_file, abstracts_file, batch_size):
            processed += len(y_train)

            print('Model is training: epoch {}/{} total samples {}'.format(e, epochs, processed))

            model.train_step(X_train, y_train, batch_size, epochs=1, validation_split=validation_split, wandb_callback=wandb_callback)

        model.save(config['model_save_dir']+'/'+config['model_name'] +'/'+ 'epoch_{}'.format(e))
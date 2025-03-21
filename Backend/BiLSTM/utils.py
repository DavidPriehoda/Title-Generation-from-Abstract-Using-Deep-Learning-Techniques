import os
import pickle
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def read_dataset(path):
    # Read in the dataset
    print('Reading dataset...')
    df = pd.read_csv(path)

    print('Extracting titles and summaries...')
    titles = df['Title'].str.replace(',', '').values.tolist()
    abstracts = df['Abstract'].str.replace(',', '').values.tolist()

    return titles, abstracts


def load_vocab_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data['abstractVocabulary'], data['titleVococabulary'], data['abstract_idx2word'], data['title_idx2word'], data['max_len'], data['title_vocab_size'], data['abstract_vocab_size']


def train_val_split(abstracts, titles, validation_split=0.2):

    # Split the data into training and validation sets
    split_index = int(len(titles) * (1 - validation_split))
    x_train = abstracts[:split_index]
    y_train = titles[:split_index]
    x_val = abstracts[split_index:]
    y_val = titles[split_index:]

    return x_train, y_train, x_val, y_val


def load_training_batch(titles_file, abstracts_file, batch_size):
    with open(titles_file, 'r') as titles_train, open(abstracts_file, 'r') as abstracts_train:
        while True:
            title_train_batch = []
            abstract_train_batch = []

            for _ in range(batch_size):
                title_line = titles_train.readline().strip()
                abstract_line = abstracts_train.readline().strip()
                if not title_line or not abstract_line:
                    break
                title_train_batch.append([int(x) for x in title_line.split(' ')])
                abstract_train_batch.append([int(x) for x in abstract_line.split(' ')])

            if not title_train_batch or not abstract_train_batch:
                break
            
            yield abstract_train_batch, title_train_batch


def create_model_folder(dir, model_name):
    if not os.path.exists(dir):
        os.makedirs(dir)
    if not os.path.exists(dir+'/'+model_name):
        os.makedirs(dir+'/'+model_name)

def clean_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    text = text.replace('\n', '')
    text = text.replace('\t', '')
    text = text.replace('\r', '')
    return text

stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    words = word_tokenize(text)
    text = ' '.join([w for w in words if not w in stop_words])
    return text
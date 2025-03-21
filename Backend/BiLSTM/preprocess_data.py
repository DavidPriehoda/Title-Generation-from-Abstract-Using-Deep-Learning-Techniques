import os
import pickle
import operator

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils import read_dataset
from config import config


def preprocess_data(titles, abstracts, max_len, vocab_size_titles, vocab_size_abstracts):

    print('Preprocessing data...')

    #truncate the abstracts and titles to max_len and add start and end tokens
    abstracts = [' '.join(a.split()[:max_len]) for a in abstracts]
    titles = [' '.join(t.split()[:max_len-2]) for t in titles]

    titles = ['<start> ' + t + ' <end>' for t in titles]


    print('Creating abstract vocabulary...')

    tokenizer_abstracts = Tokenizer()

    # fit the tokenizer on the abstracts
    tokenizer_abstracts.fit_on_texts(abstracts)
    
    # get the word counts for each word
    counts_abstracts = dict(tokenizer_abstracts.word_counts)

    # sort the words by their counts
    words_1 = sorted(counts_abstracts.items(),
                     key=operator.itemgetter(1), reverse=True)
    
    # get the top vocab_size words
    vocabWord_1 = [x[0] for x in words_1][:vocab_size_abstracts]
    
    # add the zero and unknown tokens
    vocabWord_1.insert(0, '<zero>')
    vocabWord_1.insert(1, '<unk>')
    
    # create a list of indices
    vocabIdx_1 = list(range(0, vocab_size_abstracts))
    
    # create a dictionary mapping words to indices
    abstractVocabulary = dict(zip(vocabWord_1, vocabIdx_1))

    
    print('Creating title vocabulary...')
    # Repeat the same process for the title vocabulary
    tokenizer_titles = Tokenizer()
    tokenizer_titles.fit_on_texts(titles)
    counts_2 = dict(tokenizer_titles.word_counts)
    words_2 = sorted(counts_2.items(),
                     key=operator.itemgetter(1), reverse=True)
    vocabWord_2 = [x[0] for x in words_2][:vocab_size_titles]
    vocabWord_2.insert(0, '<zero>')
    vocabWord_2.insert(1, '<unk>')
    vocabWord_2.insert(2, '<start>')
    vocabWord_2.insert(3, '<end>')
    vocabIdx_2 = list(range(0, vocab_size_titles))
    titleVococabulary = dict(zip(vocabWord_2, vocabIdx_2))

    print('Mapping words to indices...')
    # Map the words in the abstracts to their indices
    unk_idx_abstract = abstractVocabulary.get('<unk>')
    abstractData = []
    for i in range(len(abstracts)):
        x = text_to_word_sequence(
            abstracts[i], filters='', lower=True, split=" ")
        d = [abstractVocabulary[n]
             if n in abstractVocabulary else unk_idx_abstract for n in x]
        abstractData.append(d)

    # Map the words in the titles to their indices
    unk_idx_title = titleVococabulary.get('<unk>')
    titleData = []
    for i in range(len(titles)):
        x = text_to_word_sequence(titles[i], filters='', lower=True, split=" ")
        d = [titleVococabulary[n]
             if n in titleVococabulary else unk_idx_title for n in x]
        titleData.append(d)

    print('Padding sequences...')
    # Pad the abstracts and titles to the same length
    abstractData = pad_sequences(abstractData, maxlen=max_len, padding='post')
    titleData = pad_sequences(titleData, maxlen=max_len, padding='post')

    # Create a dictionary mapping indices to words
    abstract_idx2word = {v: k for k, v in abstractVocabulary.items()}
    title_idx2word = {v: k for k, v in titleVococabulary.items()}

    return abstractData, titleData, abstractVocabulary, titleVococabulary, abstract_idx2word, title_idx2word


def save_data(abstracts_train, titles_train, abstractVocabulary, titleVococabulary, abstract_idx2word, title_idx2word):
    print('Saving data to disk...')

    # Save the vocabulary and other data for later use in the model training
    preprocessed_data = {
        'abstractVocabulary': abstractVocabulary,
        'titleVococabulary': titleVococabulary,
        'abstract_idx2word': abstract_idx2word,
        'title_idx2word': title_idx2word,
        'max_len': max_len,
        'title_vocab_size': len(titleVococabulary),
        'abstract_vocab_size': len(abstractVocabulary)}

    with open(save_dir + '/vocab.pkl', 'wb') as f:
        pickle.dump(preprocessed_data, f)

    # Save the abstract and title data
    np.savetxt(save_dir + '/abstracts.txt', abstracts_train, fmt='%d')
    np.savetxt(save_dir + '/titles.txt', titles_train, fmt='%d')


if __name__ == '__main__':
    max_len = config['max_len']
    vocab_size_titles = config['vocab_size_titles']
    vocab_size_abstracts = config['vocab_size_abstracts']
    dataset = config['dataset_dir'] + config['dataset_name']
    save_dir = config['preprocessed_save_dir'] + config['model_name']

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    titles, abstracts = read_dataset(dataset)

    save_data(*preprocess_data(titles, abstracts, max_len, vocab_size_titles, vocab_size_abstracts))

    print('Output saved to ' + save_dir)
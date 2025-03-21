import tensorflow as tf
from tensorflow.keras.models import load_model
from utils import clean_text, remove_stopwords, load_vocab_data
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from math import log
import random

class Generator:
    def __init__(self, model_path, vocab_path):
        self.model = load_model(model_path)
        self.word2idx_abstract, _, self.idx2word_abstract, self.idx2word_title, self.max_len, _, _ = load_vocab_data(vocab_path)
    
    def generate(self, abstract):
        abstract = clean_text(abstract)
        abstract = remove_stopwords(abstract)
        abstract = self.tokenize(abstract, self.word2idx_abstract, self.max_len)
        generated_titles = self.model.predict(abstract)
        return generated_titles
    
    def beam_search(self, data, k, temperature, beam_width):
        sequences = [[list(), set(), 0.0]]
        for row in data:
            all_candidates = []
            for i, (seq, used_words, score) in enumerate(sequences):
                if seq and seq[-1] == 3:  # stop adding words if the last token is 3
                    all_candidates.append([seq, used_words, score])
                    continue

                # Compute the adjusted probabilities with temperature
                adjusted_probs = np.log(row) / temperature
                adjusted_probs = np.exp(adjusted_probs) / np.sum(np.exp(adjusted_probs))

                # Find the top beam_width words to consider
                top_words = np.argsort(adjusted_probs)[::-1][:beam_width]

                for j in top_words:
                    if j in used_words:
                        continue
                    
                    if j == 1 or (j==3 and random.random() > 0.5): #encourage longer titles 
                        # Find the index of the next most likely word
                        next_best_word = np.argsort(row)[-2]

                        attempted_words = set()
                        while next_best_word in used_words or next_best_word == 1 or next_best_word == 3:
                            next_best_word = np.argsort(row)[-(np.where(np.argsort(row) != next_best_word)[0][-2])]
                            if next_best_word in attempted_words:
                                break  # tried all possible words
                            attempted_words.add(next_best_word)
                        candidate_score = score - log(adjusted_probs[next_best_word])
                        candidate = [seq + [next_best_word], used_words.union({next_best_word}), candidate_score]
                    else:
                        candidate_score = score - log(adjusted_probs[j])
                        candidate = [seq + [j], used_words.union({j}), candidate_score]
                    all_candidates.append(candidate)
            
            # Find the top k candidates
            ordered = sorted(all_candidates, key=lambda tup: tup[2])
            sequences = ordered[:k]

        return sequences

    def tokenize(self, abstract, word2idx_abstract, max_len):
        abstract_tokens = []
        for word in abstract.split():
            if word in word2idx_abstract:
                abstract_tokens.append(word2idx_abstract[word])
            else:
                abstract_tokens.append(word2idx_abstract['<unk>'])
        
        padded_abstract_tokens = pad_sequences([abstract_tokens], maxlen=max_len, padding='post', truncating='post')
        return padded_abstract_tokens

    def __call__(self, abstract, num_return_sequences=1, temperature=1, beam_width=32):
        """
        Predict the title of a given abstract.
        """
        generated_titles = self.generate(abstract)

        beam_search_results = self.beam_search(generated_titles[0], num_return_sequences, temperature, beam_width)
        
        titles = []
        
        for sequence in beam_search_results:
            title = ' '.join([self.idx2word_title[i] for i in sequence[0] if i != 0 and i != 2 and i != 3])
            titles.append(title) if title not in titles else None
        
        return titles
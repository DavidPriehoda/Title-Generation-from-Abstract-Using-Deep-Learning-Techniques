import pandas as pd
from random import sample

import nltk
from nltk.translate import bleu_score, meteor_score
from nltk.translate.bleu_score import SmoothingFunction
from rouge_score import rouge_scorer

from generator import Generator

def evaluate_generated_titles(ground_truth, generated_titles, temperature, beam_width):

    # Tokenize titles for BLEU and METEOR scores
    ground_truth_tokens = [nltk.word_tokenize(title.lower()) for title in ground_truth]
    generated_titles_tokens = [nltk.word_tokenize(title.lower()) for title in generated_titles]

    bleu_scores = []
    meteor_scores = []
    rouge_scores = []

    smoothing = SmoothingFunction().method1
    rouge_l_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    # Calculate BLEU and METEOR scores
    for reference, candidate in zip(ground_truth_tokens, generated_titles_tokens):
        bleu_scores.append(bleu_score.sentence_bleu([reference], candidate, smoothing_function=smoothing))
        meteor_scores.append(meteor_score.single_meteor_score(reference, candidate))

    # Calculate ROUGE-L
    for reference, candidate in zip(ground_truth, generated_titles):
        rouge_scores.append(rouge_l_scorer.score(reference, candidate)['rougeL'].fmeasure)
    
    avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
    avg_meteor_score = sum(meteor_scores) / len(meteor_scores)
    avg_rouge_l = sum(rouge_scores) / len(rouge_scores)

    print("-"*25)
    print(f"temp: {temperature}, beam: {beam_width}")
    print()
    print(f"Average ROUGE-L score: {avg_rouge_l:.4f}")
    print(f"Average BLEU score: {avg_bleu_score:.4f}")
    print(f"Average METEOR score: {avg_meteor_score:.4f}")
    print("-"*25)

def read_dataset(path):
    # Read in the dataset
    print('Reading dataset...')
    df = pd.read_csv(path)

    print('Extracting titles and abstracts...')
    titles = df['Title'].str.replace(',', '').values.tolist()
    abstracts = df['Abstract'].str.replace(',', '').values.tolist()

    return titles, abstracts

if __name__ == '__main__':

    model_path = './models/model-t5/checkpoint-5000'

    dataset = input("Test Dataset path: ")
    num_test_samples = int(input("Number of test samples: "))

    test_vals = {"temperature":[1,5,10,32], "beam_width":[1,5,16,32]}


    titles, abstracts = read_dataset(dataset)

    #randomly chose titles and abstracts for testing
    rand_idx = sample(range(len(titles)), num_test_samples)
    ground_truth_titles = [titles[i] for i in rand_idx]
    abstracts = [abstracts[i] for i in rand_idx]

    model = Generator(model_path)

    for num_beams in test_vals["beam_width"]:
        for temperature in test_vals["temperature"]:

            generated_titles = []
            # Generate titles
            for i,abstract in enumerate(abstracts):
                generated_titles += model(abstract, temperature=temperature, num_beams=num_beams)

                print(f"Generating titles... {i+1}/{num_test_samples}", end='\r')

            evaluate_generated_titles(ground_truth_titles, generated_titles, temperature, num_beams)
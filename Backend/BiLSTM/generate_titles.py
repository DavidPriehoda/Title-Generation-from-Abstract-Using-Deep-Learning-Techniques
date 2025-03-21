from generator import Generator

if __name__ == '__main__':

    model_path = './models/BiLSTM_model_6/epoch_200'
    vocab_path = './preprocessed/BiLSTM_model_6/vocab.pkl'

    abstract = input("Enter abstract: ")
    num_return_sequences = int(input("Enter number of titles to generate: "))
    temperature = float(input("Enter temperature: "))
    beam_width = int(input("Enter beam width: "))

    gen = Generator(model_path, vocab_path)

    titles = gen(abstract, num_return_sequences, temperature, beam_width)

    for t in titles:
        print(t,"\n")

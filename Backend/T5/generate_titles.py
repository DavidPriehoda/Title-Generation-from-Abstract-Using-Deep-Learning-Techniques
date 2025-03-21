from generator import Generator

if __name__ == '__main__':

    model_path = 'models/model-t5/checkpoint-5000'

    abstract = input("Enter abstract: ")
    num_return_sequences = int(input("Enter number of titles to generate: "))
    temperature = float(input("Enter temperature: "))
    beam_width = int(input("Enter beam width: "))


    gen = Generator(model_path)

    titles = gen(abstract, num_return_sequences, temperature, beam_width)

    for t in titles:
        print(t,"\n")
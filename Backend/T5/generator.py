from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Generator:
    def __init__(self, model_path, temperature=25, num_beams=32, max_gen_length=128):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        self.temperature = temperature
        self.num_beams = num_beams
        self.max_gen_length = max_gen_length
        self.n_titles = 1
        
    def preprocess(self, abstract):
        return self.tokenizer([abstract], max_length=512, return_tensors='pt')
    
    def generate(self, inputs):
        return self.model.generate(
            inputs['input_ids'], 
            num_beams=self.num_beams, 
            temperature=self.temperature, 
            max_length=self.max_gen_length, 
            early_stopping=True,
            num_return_sequences=self.n_titles
        )
    
    def post_process(self, output):
        return [self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False) for ids in output]
    
    def __call__(self, abstract, n_titles=1, temperature=1, num_beams=32, max_gen_length=128):
        
        self.n_titles = n_titles
        self.temperature = temperature
        self.num_beams = num_beams
        self.max_gen_length = max_gen_length

        inputs = self.preprocess(abstract)
        output = self.generate(inputs)
        titles = self.post_process(output)
        
        return titles

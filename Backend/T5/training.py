import numpy as np
import warnings
import logging
import nltk
from nltk.tokenize import sent_tokenize
import wandb
from datasets import load_from_disk, load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

from config import *

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


nltk.download('punkt')


tokenizer = AutoTokenizer.from_pretrained('google/t5-v1_1-base')
dataset = load_from_disk('preprocessed')

def preprocess_data(data):
    dataset = tokenizer(data['Abstract'], max_length=MAX_ABSTRACT_LEN, padding=True, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(data['Title'], max_length=MAX_TITLE_LEN, padding=True, truncation=True)
    
    labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]
    dataset['labels'] = labels["input_ids"]
    return dataset


processed_dataset = dataset.map(
    preprocess_data,
    batched=True,
    remove_columns=['Abstract', 'Title'],
    desc="Running tokenizer on dataset",
)

metric = load_metric("rouge")

### This function is taken from the HuggingFace course
### https://huggingface.co/learn/nlp-course/chapter7/5#metrics-for-text-summarization
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Decode generated summaries into text
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # ROUGE expects a newline after each sentence
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
    # Compute ROUGE scores
    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract the median scores
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}


training_args = Seq2SeqTrainingArguments(
    output_dir=f"{model_save_dir}model-t5",
    evaluation_strategy="steps",
    eval_steps=eval_every,
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=weight_decay,
    save_steps=500,
    save_total_limit=3,
    num_train_epochs=num_epochs,
    predict_with_generate=True,
    logging_steps=log_every,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
    resume_from_checkpoint=True,
    gradient_accumulation_steps=gradient_accumulation_steps,
    fp16=True,
    report_to="wandb"
)

model = AutoModelForSeq2SeqLM.from_pretrained('google/t5-v1_1-base')

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

wandb.init(project="t5-transformer")


trainer = Seq2SeqTrainer(
    model,
    training_args,
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()


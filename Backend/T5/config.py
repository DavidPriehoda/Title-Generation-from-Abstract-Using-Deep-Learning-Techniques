MAX_ABSTRACT_LEN = 512
MAX_TITLE_LEN = 128

batch_size = 8
num_epochs = 4
learning_rate = 5.6e-5
weight_decay = 0.01
log_every = 50
eval_every = 1000
gradient_accumulation_steps = 3
lr_scheduler_type = "linear"

dataset_dir = '../dataset/'
model_save_dir = './models/'
dataset_name = 'dataset.csv'
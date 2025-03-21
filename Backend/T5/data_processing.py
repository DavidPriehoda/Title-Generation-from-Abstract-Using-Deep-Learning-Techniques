import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from config import *


papers = pd.read_csv(dataset_dir + dataset_name)
papers = papers.drop('Category', axis=1)

train, test = train_test_split(papers,test_size=0.2)

dataset_train = Dataset.from_pandas(train)
dataset_validation = Dataset.from_pandas(test)

ds = DatasetDict()
ds['train'] = dataset_train
ds['validation'] = dataset_validation


ds.save_to_disk('preprocessed')
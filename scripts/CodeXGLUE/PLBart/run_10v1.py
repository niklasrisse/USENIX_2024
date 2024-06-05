import os
import sys
import random
import pickle
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from typing import List
import argparse

from clang import *
from clang import cindex

import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

import sklearn
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from transformers import PLBartForSequenceClassification, AutoTokenizer
from transformers import get_scheduler

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from transformations import *

STR_TO_TRAFO = {
    "no_transformation" : no_transformation,
    "tf_1" : tf_1,
    "tf_2" : tf_2,
    "tf_3" : tf_3,
    "tf_4" : tf_4,
    "tf_5" : tf_5,
    "tf_6" : tf_6,
    "tf_7" : tf_7,
    "tf_8" : tf_8,
    "tf_9" : tf_9,
    "tf_10" : tf_10,
    "tf_11" : tf_11,
    "tf_12" : tf_12,
    "tf_13" : tf_13,
    }

parser = argparse.ArgumentParser()
parser.add_argument("--trafo", required=True)
parser.add_argument("--random", default=False, required=False)
args = parser.parse_args()
trafo_from_args = STR_TO_TRAFO[args.trafo]
random_guessing = args.random == "True"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
print(device)

# deterministic settings for exact reproducibility
seedlist = [42, 834, 692, 489, 901, 408, 819, 808, 531, 166]
fixed_seed = 834 #random.choice(seedlist)

os.environ['PYTHONHASHSEED'] = str(fixed_seed)
torch.manual_seed(fixed_seed)
torch.cuda.manual_seed(fixed_seed)
torch.cuda.manual_seed_all(fixed_seed)
np.random.seed(fixed_seed)
random.seed(fixed_seed)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

machine = "localhost" if os.path.dirname(os.path.realpath(__file__)).split("/")[1] == "Users" else "cluster"

if machine == "cluster":
    pathprefix = "USENIX_2024/"
else:
    pathprefix = ""

my_tokenizer = AutoTokenizer.from_pretrained("uclanlp/plbart-c-cpp-defect-detection")


train_index=set()
valid_index=set()
test_index=set()

with open('./' + pathprefix + 'datasets/CodeXGLUE/train-IDs.txt') as f:
    for line in f:
        line=line.strip()
        train_index.add(int(line))

with open('./' + pathprefix + 'datasets/CodeXGLUE/valid-IDs.txt') as f:
    for line in f:
        line=line.strip()
        valid_index.add(int(line))

with open('./' + pathprefix + 'datasets/CodeXGLUE/test-IDs.txt') as f:
    for line in f:
        line=line.strip()
        test_index.add(int(line))


mydata = pd.read_json('./' + pathprefix + 'datasets/CodeXGLUE/CodeXGLUE.json')
developer_commit_data_test = pd.read_json('./' + pathprefix + 'datasets/VulnPatchPairs/VulnPatchPairs-Test.json')

m1=mydata.iloc[list(train_index)]
m2=mydata.iloc[list(valid_index)]
m3=mydata.iloc[list(test_index)]
m4 = developer_commit_data_test


def encodeDataframe(df):
    
    encodings = my_tokenizer.batch_encode_plus(df.func, max_length=1024, padding='max_length', truncation=True)
    
    return encodings, df.target.tolist()

class MyCustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        assert len(self.encodings['input_ids']) == len(self.encodings['attention_mask']) ==  len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def apply_transformation(df, all_trafos, trafo_to_apply, training_set_sample_neg, training_set_sample_pos, trafo_not_to_apply_in_random_selection = None):
    
    if trafo_to_apply.__name__ == "tf_11":
        df.func = df.func.apply(trafo_to_apply, args=(all_trafos, training_set_sample_neg, training_set_sample_pos, trafo_not_to_apply_in_random_selection))
    elif trafo_to_apply.__name__ == "tf_10":
        df.func = df.func.apply(trafo_to_apply, args=(training_set_sample_neg,))
    elif trafo_to_apply.__name__ == "tf_13":
        df.func = df.func.apply(trafo_to_apply, args=(training_set_sample_pos,))
    else:
        df.func = df.func.apply(trafo_to_apply)
        
    return df

    
TF_TO_APPLY_ON_TRAIN = tf_11
TF_TO_APPLY_ON_TEST = trafo_from_args

semantic_preserving_transformations = [no_transformation, tf_1, tf_2, tf_3, tf_4, tf_5, tf_6, tf_7, tf_8, tf_9, tf_10, tf_11]

m1_not_vulnerable = m1[m1["target"] == 0]
training_set_sample_neg = m1_not_vulnerable.iloc[0]["func"]

m1_vulnerable = m1[m1["target"] == 1]
training_set_sample_pos = m1_vulnerable.iloc[0]["func"]
    
m1 = apply_transformation(m1, semantic_preserving_transformations, TF_TO_APPLY_ON_TRAIN, training_set_sample_neg, training_set_sample_pos, trafo_from_args)

TRAINING_BATCH_SIZE = 8

train_encodings, train_labels = encodeDataframe(m1)
test_encodings, test_labels = encodeDataframe(m3)

train_dataset = MyCustomDataset(train_encodings, train_labels)
test_dataset = MyCustomDataset(test_encodings, test_labels)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=TRAINING_BATCH_SIZE)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=4)

print(len(train_dataloader))
print(len(test_dataloader))

num_epochs = 10
num_training_steps = num_epochs * len(train_dataloader)
num_testing_steps = num_epochs * len(test_dataloader)

model = PLBartForSequenceClassification.from_pretrained("uclanlp/plbart-base")
model.to(device)

print(model)

optimizer = AdamW(model.parameters(), lr=3e-6)
lr_scheduler = get_scheduler(
     name="linear", optimizer=optimizer, num_warmup_steps=2500, num_training_steps=num_training_steps
)

criterion = torch.nn.CrossEntropyLoss() 
criterion.to(device)

results = dict()

progress_bar = tqdm(range(num_training_steps + num_testing_steps))

for epoch in range(num_epochs):
    
    model.train()
    
    losses = []
    predictions = []
    labels = []
    
    for batch in train_dataloader:
        
        batch = {k: v.to(device) for k, v in batch.items()}
        
        if not random_guessing:
            optimizer.zero_grad()
            
            outputs = model(**batch)
            loss = criterion(outputs.logits, batch["labels"])
        
            loss.backward()
            
            logits = outputs.logits
            
            losses += [loss.item()]
            
            predictions += torch.argmax(logits, dim=-1).tolist()
            labels += batch["labels"].tolist()

            optimizer.step()
            lr_scheduler.step()
        
        progress_bar.update(1)
    
    model.eval()

        
    if machine == "cluster":
        m3=mydata.iloc[list(test_index)]
    else:
        m3=mydata.iloc[list(test_index)[:8]]
    
    m3 = apply_transformation(m3, semantic_preserving_transformations, TF_TO_APPLY_ON_TEST, training_set_sample_neg, training_set_sample_pos, TF_TO_APPLY_ON_TEST)

    test_encodings, test_labels = encodeDataframe(m3)

    test_dataset = MyCustomDataset(test_encodings, test_labels)

    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=4)
    
    predictions = []
    labels = []

    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        if random_guessing:
            predictions += len(batch["labels"].tolist()) * [random.randint(0, 1)]
            labels += batch["labels"].tolist()
        else:
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            
            predictions += torch.argmax(logits, dim=-1).tolist()
            labels += batch["labels"].tolist()
        
        progress_bar.update(1)
    
    test_accuracy = accuracy_score(labels, predictions)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(labels, predictions, average='binary', zero_division=1)
    test_tn, test_fp, test_fn, test_tp = confusion_matrix(labels, predictions).ravel()
    
    if epoch == 0:
        results[TF_TO_APPLY_ON_TEST.__name__] = dict()
    results[TF_TO_APPLY_ON_TEST.__name__][epoch] = dict()
    results[TF_TO_APPLY_ON_TEST.__name__][epoch]["epoch"] = epoch
    results[TF_TO_APPLY_ON_TEST.__name__][epoch]["train/loss"] = np.array(losses).mean()
    results[TF_TO_APPLY_ON_TEST.__name__][epoch]["test/accuracy"] = test_accuracy
    results[TF_TO_APPLY_ON_TEST.__name__][epoch]["test/precision"] = test_precision
    results[TF_TO_APPLY_ON_TEST.__name__][epoch]["test/recall"] = test_recall
    results[TF_TO_APPLY_ON_TEST.__name__][epoch]["test/f1"] = test_f1
        


save_name = os.path.dirname(os.path.realpath(__file__)).split("/")[-2] + "-" + os.path.dirname(os.path.realpath(__file__)).split("/")[-1] + "-" + TF_TO_APPLY_ON_TRAIN.__name__ + "-" + TF_TO_APPLY_ON_TEST.__name__ 
    
with open('./' + pathprefix + 'results/{}.pkl'.format(save_name), 'wb') as f:
    pickle.dump(results, f)


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
import wandb

from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer
from transformers import get_constant_schedule
from transformers import get_linear_schedule_with_warmup
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
print(device)


# deterministic settings for exact reproducibility
seedlist = [42, 834, 692, 489, 901, 408, 819, 808, 531, 166]

NUM_EPOCHS = 20
TRAINING_SET_FRAC = 1.0
RANDOM_SEED = 531
LEARNING_RATE = 3e-5
BATCH_SIZE = 8
SCHEDULER_NAME = "constant"
NUM_SCHEDULER_RESTARTS = 0
SCHEDULER_WARMUP_PERCENTAGE = 0

os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.environ['WANDB_MODE'] = 'offline'

machine = "localhost" if os.path.dirname(os.path.realpath(__file__)).split("/")[1] == "Users" else "cluster"

if machine == "cluster":
    pathprefix = "USENIX_2024/"
else:
    pathprefix = ""

tokenizer = T5Tokenizer.from_pretrained('razent/cotext-1-ccg')

ids_neg = tokenizer.encode('negative </s>')
ids_pos = tokenizer.encode('positive </s>')

vpp_train = pd.read_json('./' + pathprefix + 'datasets/VulnPatchPairs/VulnPatchPairs-Train.json')
vpp_valid = pd.read_json('./' + pathprefix + 'datasets/VulnPatchPairs/VulnPatchPairs-Valid.json')
vpp_test = pd.read_json('./' + pathprefix + 'datasets/VulnPatchPairs/VulnPatchPairs-Test.json')

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

codexglue = pd.read_json('./' + pathprefix + 'datasets/CodeXGLUE/CodeXGLUE.json')

codexglue_train = codexglue.iloc[list(train_index)]
codexglue_valid = codexglue.iloc[list(valid_index)]
codexglue_test = codexglue.iloc[list(test_index)]

def prependTaskDesc(value):
    
    return "defect_detection: " + value

def toStrTargets(value):
    
    assert value == "0" or value == "1"
    
    if value == "0":
        return "negative </s>"
    return "positive </s>"

def cotext_preprop(code):
    
    code = code.replace("\n", " NEW_LINE ")
    code = code.replace("}", " CLOSE_CURLY_TOKEN ")
    code = code.replace("{", " OPEN_CURLY_TOKEN ")
    code = code.replace("]", " CLOSE_SQUARE_TOKEN ")
    code = code.replace("[", " OPEN_SQUARE_TOKEN ")
    code = code.replace("<", " SMALLER_TOKEN ")
    code = code.replace(">", " GREATER_TOKEN ")
    code = code.replace("#", " SHARP_TOKEN ")
    
    return code
    

def prepareDataframe(df):
    
    df['target'] = df['target'].map(str)
    df['target'] = df['target'].map(toStrTargets)
    df['func'] = df['func'].map(prependTaskDesc)
    
    return df

def encodeDataframe(df):
    
    encodings = []
    labels = []

    for row in df.itertuples(index=True, name='Pandas'):
    
        enc = tokenizer.batch_encode_plus([row.func], max_length=1024, padding='max_length', truncation=True, return_tensors="pt")
        lab = tokenizer.batch_encode_plus([row.target], max_length=2, padding='max_length', truncation=True, return_tensors="pt")
        
        encodings.append(enc)
        labels.append(lab)
        
    
    return encodings, labels

class MyCustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, index):
        source_ids = self.encodings[index]["input_ids"].squeeze()
        target_ids = self.labels[index]["input_ids"].squeeze()

        src_mask    = self.encodings[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.labels[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

    def __len__(self):
        return len(self.labels)

def separate_vulns_and_patches(df):
    
    patches = df.drop('vulnerable_func', axis=1)
    patches = patches.rename(columns={'patched_func': 'func'})
    patches["target"] = 0
    
    vulns = df.drop('patched_func', axis=1)
    vulns = vulns.rename(columns={'vulnerable_func': 'func'})
    vulns["target"] = 1
    
    fused_df = patches.append(vulns)
    fused_df = fused_df.reset_index()
    
    return fused_df

def get_scheduler(optim, name, num_restarts, num_training_steps, warmup_percentage):
    
    if name == "linear":
        num_warmup_steps = int(num_training_steps * warmup_percentage)
        return get_linear_schedule_with_warmup(optimizer=optim, num_training_steps=num_training_steps, num_warmup_steps=num_warmup_steps)
    elif name == "cosine_with_restarts":
        num_warmup_steps = int(num_training_steps * warmup_percentage / (num_restarts + 1))
        return get_cosine_with_hard_restarts_schedule_with_warmup(optimizer=optim, num_training_steps=num_training_steps, num_warmup_steps=num_warmup_steps, num_cycles=num_restarts)
    elif name == "constant":
        return get_constant_schedule(optimizer=optim)
    else:
        raise Exception("schedule not available")

vpp_train = vpp_train.sample(frac=TRAINING_SET_FRAC)
vpp_test = vpp_test.sample(frac=1.0)

vpp_train = separate_vulns_and_patches(vpp_train)
vpp_valid = separate_vulns_and_patches(vpp_valid)
vpp_test = separate_vulns_and_patches(vpp_test)

if machine == "localhost":
    vpp_train = vpp_train.iloc[:5]
    vpp_valid = vpp_valid.iloc[:5]
    vpp_test = vpp_test.iloc[:5]
    codexglue_valid = codexglue_valid.iloc[:5]
    codexglue_test = codexglue_test.iloc[:5]

vpp_train.func = vpp_train.func.apply(cotext_preprop)
vpp_valid.func = vpp_valid.func.apply(cotext_preprop)
vpp_test.func = vpp_test.func.apply(cotext_preprop)
codexglue_valid.func = codexglue_valid.func.apply(cotext_preprop)
codexglue_test.func = codexglue_test.func.apply(cotext_preprop)

vpp_train = prepareDataframe(vpp_train)
vpp_valid = prepareDataframe(vpp_valid)
vpp_test = prepareDataframe(vpp_test)
codexglue_valid = prepareDataframe(codexglue_valid)
codexglue_test = prepareDataframe(codexglue_test)

vpp_train_encodings, vpp_train_labels = encodeDataframe(vpp_train)
vpp_valid_encodings, vpp_valid_labels = encodeDataframe(vpp_valid)
vpp_test_encodings, vpp_test_labels = encodeDataframe(vpp_test)
codexglue_valid_encodings, codexglue_valid_labels = encodeDataframe(codexglue_valid)
codexglue_test_encodings, codexglue_test_labels = encodeDataframe(codexglue_test)

vpp_train_dataset = MyCustomDataset(vpp_train_encodings, vpp_train_labels)
vpp_valid_dataset = MyCustomDataset(vpp_valid_encodings, vpp_valid_labels)
vpp_test_dataset = MyCustomDataset(vpp_test_encodings, vpp_test_labels)
codexglue_valid_dataset = MyCustomDataset(codexglue_valid_encodings, codexglue_valid_labels)
codexglue_test_dataset = MyCustomDataset(codexglue_test_encodings, codexglue_test_labels)

vpp_train_dataloader = DataLoader(vpp_train_dataset, shuffle=True, batch_size=BATCH_SIZE)
vpp_valid_dataloader = DataLoader(vpp_valid_dataset, shuffle=False, batch_size=BATCH_SIZE)
vpp_test_dataloader = DataLoader(vpp_test_dataset, shuffle=False, batch_size=BATCH_SIZE)
codexglue_valid_dataloader = DataLoader(codexglue_valid_dataset, shuffle=False, batch_size=BATCH_SIZE)
codexglue_test_dataloader = DataLoader(codexglue_test_dataset, shuffle=False, batch_size=BATCH_SIZE)

print(len(vpp_train_dataloader))
print(len(vpp_valid_dataloader))
print(len(vpp_test_dataloader))
print(len(codexglue_valid_dataloader))
print(len(codexglue_test_dataloader))

num_training_steps = NUM_EPOCHS * len(vpp_train_dataloader)
num_validation_steps = NUM_EPOCHS * len(codexglue_valid_dataloader)
num_testing_steps = NUM_EPOCHS * len(codexglue_test_dataloader) + NUM_EPOCHS * len(vpp_test_dataloader)

model = T5ForConditionalGeneration.from_pretrained('razent/cotext-1-ccg')
model.to(device)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

lr_scheduler = get_scheduler(optim=optimizer, name=SCHEDULER_NAME, num_restarts=NUM_SCHEDULER_RESTARTS, num_training_steps=num_training_steps, warmup_percentage=SCHEDULER_WARMUP_PERCENTAGE)

results = dict()

progress_bar = tqdm(range(num_training_steps + num_testing_steps))

for epoch in range(NUM_EPOCHS):
    
    model.train()
    
    losses = []
    predictions = []
    labels = []
    
    for batch in vpp_train_dataloader:
        
        batch = {k: v for k, v in batch.items()}
        
        optimizer.zero_grad()
        
        target_ids = batch["target_ids"]
        target_ids[target_ids == tokenizer.pad_token_id] = -100
        
        outputs = model(input_ids=batch["source_ids"].to(device), attention_mask=batch["source_mask"].to(device), labels=target_ids.to(device))
        
        loss = outputs.loss
    
        loss.backward()
        
        losses += [loss.item()]

        optimizer.step()
        lr_scheduler.step()
        
        progress_bar.update(1)
    
    model.eval()
    
    predictions = []
    labels = []

    for batch in vpp_test_dataloader:
        batch = {k: v for k, v in batch.items()}
        
        with torch.no_grad():
            outputs = model.generate(input_ids=batch["source_ids"].to(device), attention_mask=batch["source_mask"].to(device), max_length=5)
    
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(batch["target_ids"].to(device), skip_special_tokens=True)
        
        labels_as_numbers = [1 if d == "positive" else 0 for d in decoded_labels]
        
        predictions_as_numbers = []
        
        for i, output in enumerate(decoded_outputs):
            if output == "positive":
                predictions_as_numbers.append(1)
            elif output == "negative":
                predictions_as_numbers.append(0)
            else:
                predictions_as_numbers.append(random.randint(0,1))
        
        labels += labels_as_numbers
        predictions += predictions_as_numbers
        
        progress_bar.update(1)
    
    test_accuracy = accuracy_score(labels, predictions)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(labels, predictions, average='binary', zero_division=1)
    
    predictions_codexglue = []
    labels_codexglue = []

    for batch in codexglue_test_dataloader:
        batch = {k: v for k, v in batch.items()}
        
        with torch.no_grad():
            outputs = model.generate(input_ids=batch["source_ids"].to(device), attention_mask=batch["source_mask"].to(device), max_length=5)
    
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(batch["target_ids"].to(device), skip_special_tokens=True)
        
        labels_as_numbers = [1 if d == "positive" else 0 for d in decoded_labels]
        
        predictions_as_numbers = []
        
        for i, output in enumerate(decoded_outputs):
            if output == "positive":
                predictions_as_numbers.append(1)
            elif output == "negative":
                predictions_as_numbers.append(0)
            else:
                predictions_as_numbers.append(random.randint(0,1))
        
        labels_codexglue += labels_as_numbers
        predictions_codexglue += predictions_as_numbers
        
        progress_bar.update(1)
    
    codexglue_test_accuracy = accuracy_score(labels_codexglue, predictions_codexglue)
    codexglue_test_precision, codexglue_test_recall, codexglue_test_f1, _ = precision_recall_fscore_support(labels_codexglue, predictions_codexglue, average='binary', zero_division=1)
    
    results[epoch] = dict()
    results[epoch]["epoch"] = epoch
    results[epoch]["train/loss"] = np.array(losses).mean()
    results[epoch]["vpp/test/accuracy"] = test_accuracy
    results[epoch]["vpp/test/f1"] = test_f1
    results[epoch]["vpp/test/precision"] = test_precision
    results[epoch]["vpp/test/recall"] = test_recall
    results[epoch]["codexglue/test/accuracy"] = codexglue_test_accuracy
    results[epoch]["codexglue/test/f1"] = codexglue_test_f1
    results[epoch]["codexglue/test/precision"] = codexglue_test_precision
    results[epoch]["codexglue/test/recall"] = codexglue_test_recall
        

save_name = os.path.dirname(os.path.realpath(__file__)).split("/")[-2] + "-" + os.path.dirname(os.path.realpath(__file__)).split("/")[-1]
    
with open('./' + pathprefix + 'results/{}.pkl'.format(save_name), 'wb') as f:
    pickle.dump(results, f)



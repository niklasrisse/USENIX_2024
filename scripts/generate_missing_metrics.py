import argparse
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import inspect
import pickle
from tqdm import tqdm
import os
import seaborn as sns
import random


DATASETS = ["CodeXGLUE", "VulDeePecker"]
SEMANTIC_PRESERVING_TRANSFORMATIONS = ["no_transformation", "tf_1", "tf_2", "tf_3", "tf_4", "tf_5", "tf_6", "tf_7", "tf_8", "tf_9", "tf_10", "tf_11"]
TECHNIQUES = ["UniXcoder", "CoTexT", "GraphCodeBERT", "CodeBERT", "VulBERTa", "PLBart"]

np.random.seed(42)

dataset_stats = {
    'CodeXGLUE' : {
        'total_samples' : 2732,
        'positive_samples': 1255
    },
    'VulDeePecker' : {
        'total_samples' : 7880,
        'positive_samples': 816
    },
    'VulnPatchPairs' : {
        'total_samples' : 2732,
        'positive_samples': 1366
    }
}

results = dict()
for dataset in DATASETS:
    if dataset not in results:
        results[dataset] = dict()
        for technique in TECHNIQUES:
            if technique not in results[dataset]:
                results[dataset][technique] = dict()
            for transformation in SEMANTIC_PRESERVING_TRANSFORMATIONS:
                results_file_name = './results/{}-{}-{}.pkl'.format(dataset, technique, transformation)
                if os.path.isfile(results_file_name):
                    with open(results_file_name, 'rb') as handle:
                        results[dataset][technique][transformation] = pickle.load(handle)
                    if transformation in results[dataset][technique]:
                        for transformation_2 in results[dataset][technique][transformation]:
                            for epoch, data in results[dataset][technique][transformation][transformation_2].items():
                                accuracy = data['test/accuracy']
                                f1 = data['test/f1']
                                precision = data['test/precision'] # precision = TP / (TP + FP)
                                recall = data['test/recall'] # recall = TP / (TP + FN) = TP / P
                                
                                P = dataset_stats[dataset]["positive_samples"]
                                N = dataset_stats[dataset]["total_samples"] - dataset_stats[dataset]["positive_samples"]
                                TP = recall * P
                                FP = (TP / precision) - TP
                                
                                FPR = FP / (N)
                                FNR = 1 - recall
                                
                                results[dataset][technique][transformation][transformation_2][epoch]["test/FPR"] = FPR
                                results[dataset][technique][transformation][transformation_2][epoch]["test/FNR"] = FNR

PREFIXES = ["vpp/", "codexglue/"]
dataset = "VulnPatchPairs"               
results[dataset] = dict()
for technique in TECHNIQUES:
    if technique not in results[dataset]:
        results[dataset][technique] = dict()
        results_file_name = './results/{}-{}.pkl'.format(dataset, technique)
        
        if os.path.isfile(results_file_name):
            with open(results_file_name, 'rb') as handle:
                results[dataset][technique] = pickle.load(handle)
                for epoch, data in results[dataset][technique].items():
                    for prefix in PREFIXES:
                        accuracy = data['{}test/accuracy'.format(prefix)]
                        f1 = data['{}test/f1'.format(prefix)]
                        precision = data['{}test/precision'.format(prefix)] # precision = TP / (TP + FP)
                        recall = data['{}test/recall'.format(prefix)] # recall = TP / (TP + FN) = TP / P
                        
                        P = dataset_stats[dataset]["positive_samples"]
                        N = dataset_stats[dataset]["total_samples"] - dataset_stats[dataset]["positive_samples"]
                        TP = recall * P
                        FP = (TP / precision) - TP
                        
                        FPR = FP / (N)
                        FNR = 1 - recall
                        
                        if (FPR > 1):
                            break
                        
                        results[dataset][technique][epoch]["{}test/FPR".format(prefix)] = FPR
                        results[dataset][technique][epoch]["{}test/FNR".format(prefix)] = FNR
                                        

with open('./results/extended_results.pkl', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
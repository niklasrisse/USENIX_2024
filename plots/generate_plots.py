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

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['text.usetex'] = True

COLORS = ['#377eb8', '#ff7f00', '#4daf4a', '#a65628', '#984ea3', '#999999', '#9467BD','#dede00']
OUT_DIR = "./plots/plots/"
SEMANTIC_PRESERVING_TRANSFORMATIONS = ["no_transformation", "tf_1", "tf_2", "tf_3", "tf_4", "tf_5", "tf_6", "tf_7", "tf_8", "tf_9", "tf_10", "tf_11"]
TECHNIQUES = ["UniXcoder", "CoTexT", "GraphCodeBERT", "CodeBERT", "VulBERTa", "PLBart"]
METRICS = ["accuracy", "f1", "recall", "precision", "FPR", "FNR"]
DPI = 300
FIG_HEIGHT = 3.2
FIG_WIDTH = 6
X_TICK_LABELS_FONT_SIZE = 12
LEGEND_FONT_SIZE = 12
MEDIAN_LINE_COLOR = "black"
DATASET_TO_METRIC = {
    "CodeXGLUE" : "accuracy",
    "VulDeePecker" : "f1"
}

mpl.rcParams['font.size'] = LEGEND_FONT_SIZE

np.random.seed(42)

def load_results():
    
    results = dict()
    results_file_name = './results/extended_results.pkl'
    if os.path.isfile(results_file_name):
        with open(results_file_name, 'rb') as handle:
            results = pickle.load(handle)
                        
    results_file_name = './results/CodeXGLUE-VulBERTa-AdversarialTraining.pkl'
    if os.path.isfile(results_file_name):
            with open(results_file_name, 'rb') as handle:
                    results["CodeXGLUE"]["VulBERTa"]["ADV"] = pickle.load(handle)
    
    results_file_name = './results/CodeXGLUE-CodeBERT-AdversarialTraining.pkl'
    if os.path.isfile(results_file_name):
            with open(results_file_name, 'rb') as handle:
                    results["CodeXGLUE"]["CodeBERT"]["ADV"] = pickle.load(handle)
                    
    results_file_name = './results/CodeXGLUE-GraphCodeBERT-AdversarialTraining.pkl'
    if os.path.isfile(results_file_name):
            with open(results_file_name, 'rb') as handle:
                    results["CodeXGLUE"]["GraphCodeBERT"]["ADV"] = pickle.load(handle)
                    
    results_file_name = './results/CodeXGLUE-UniXcoder-AdversarialTraining.pkl'
    if os.path.isfile(results_file_name):
            with open(results_file_name, 'rb') as handle:
                    results["CodeXGLUE"]["UniXcoder"]["ADV"] = pickle.load(handle)
    
    results_file_name = './results/CodeXGLUE-CoTexT-AdversarialTraining.pkl'
    if os.path.isfile(results_file_name):
            with open(results_file_name, 'rb') as handle:
                    results["CodeXGLUE"]["CoTexT"]["ADV"] = pickle.load(handle)
                        
    results_file_name = './results/CodeXGLUE-PLBart-AdversarialTraining.pkl'
    if os.path.isfile(results_file_name):
            with open(results_file_name, 'rb') as handle:
                    results["CodeXGLUE"]["PLBart"]["ADV"] = pickle.load(handle)
                    
    for technique in ["CodeBERT", "UniXcoder", "CoTexT", "VulBERTa", "GraphCodeBERT", "PLBart"]:
        results["CodeXGLUE"][technique]["10v1"] = dict()
        for transformation in SEMANTIC_PRESERVING_TRANSFORMATIONS:
            if transformation != "tf_11":
                results_file_name = './results/CodeXGLUE-{}-tf_11-{}.pkl'.format(technique, transformation)
                if os.path.isfile(results_file_name):
                        with open(results_file_name, 'rb') as handle:
                                results["CodeXGLUE"][technique]["10v1"][transformation] = pickle.load(handle)[transformation]
        
                
    return results

def parse_args():
    parser=argparse.ArgumentParser(description="Script to generate all plots for paper.")
    args=parser.parse_args()
    return args

def reduce(array_to_reduce, metric="accuracy"):
    
    if metric in ["FPR", "FNR"]:
        return np.min(np.array(array_to_reduce))
    else:
        return np.max(np.array(array_to_reduce))
    
def get_score_list(data, metric = "accuracy", prefix=""):
        
    accuracies = []
    for epoch in range(max(data.keys()) + 1):
        accuracies.append(data[epoch]["{}test/{}".format(prefix, metric)])
        
    return accuracies
                
    
def fig_5_b():
    
    
    results = load_results()
    
    train_no_trafo_test_trafo_vulberta = []
    train_trafo_test_trafo_vulberta = []
    train_no_trafo_test_trafo_cotext = []
    train_trafo_test_trafo_cotext = []
    train_no_trafo_test_trafo_plbart = []
    train_trafo_test_trafo_plbart = []
    train_no_trafo_test_trafo_UniXcoder = []
    train_trafo_test_trafo_UniXcoder = []
    train_no_trafo_test_trafo_CodeBERT = []
    train_trafo_test_trafo_CodeBERT = []
    train_no_trafo_test_trafo_GraphCodeBERT = []
    train_trafo_test_trafo_GraphCodeBERT = []
    
    for transformation in SEMANTIC_PRESERVING_TRANSFORMATIONS:
        if transformation != "no_transformation":
            train_no_trafo_test_trafo_vulberta.append(reduce(get_score_list(results["CodeXGLUE"]["VulBERTa"]["no_transformation"][transformation])))
            train_no_trafo_test_trafo_UniXcoder.append(reduce(get_score_list(results["CodeXGLUE"]["UniXcoder"]["no_transformation"][transformation])))
            train_no_trafo_test_trafo_cotext.append(reduce(get_score_list(results["CodeXGLUE"]["CoTexT"]["no_transformation"][transformation])))
            train_no_trafo_test_trafo_plbart.append(reduce(get_score_list(results["CodeXGLUE"]["PLBart"]["no_transformation"][transformation])))
            train_no_trafo_test_trafo_CodeBERT.append(reduce(get_score_list(results["CodeXGLUE"]["CodeBERT"]["no_transformation"][transformation])))
            train_no_trafo_test_trafo_GraphCodeBERT.append(reduce(get_score_list(results["CodeXGLUE"]["GraphCodeBERT"]["no_transformation"][transformation])))
            
            train_trafo_test_trafo_cotext.append(reduce(get_score_list(results["CodeXGLUE"]["CoTexT"][transformation][transformation])))
            train_trafo_test_trafo_vulberta.append(reduce(get_score_list(results["CodeXGLUE"]["VulBERTa"][transformation][transformation])))
            if transformation in results["CodeXGLUE"]["UniXcoder"].keys():
                train_trafo_test_trafo_UniXcoder.append(reduce(get_score_list(results["CodeXGLUE"]["UniXcoder"][transformation][transformation])))
            if transformation in results["CodeXGLUE"]["PLBart"].keys():
                train_trafo_test_trafo_plbart.append(reduce(get_score_list(results["CodeXGLUE"]["PLBart"][transformation][transformation])))
            if transformation in results["CodeXGLUE"]["CodeBERT"].keys():
                train_trafo_test_trafo_CodeBERT.append(reduce(get_score_list(results["CodeXGLUE"]["CodeBERT"][transformation][transformation])))
            if transformation in results["CodeXGLUE"]["GraphCodeBERT"].keys():
                train_trafo_test_trafo_GraphCodeBERT.append(reduce(get_score_list(results["CodeXGLUE"]["GraphCodeBERT"][transformation][transformation])))


    fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH, FIG_HEIGHT))
    
    clean_accuracy = reduce(get_score_list(results["CodeXGLUE"]["UniXcoder"]["no_transformation"]["no_transformation"]))
    ax.plot(1.0, clean_accuracy, marker="*", color="gray", markersize=6.2)
    ax.axhline(clean_accuracy, color='gray', linestyle=":", linewidth=0.8)

    clean_accuracy = reduce(get_score_list(results["CodeXGLUE"]["CoTexT"]["no_transformation"]["no_transformation"]))
    ax.plot(2.0, clean_accuracy, marker="*", color="gray", markersize=6.2)
    ax.axhline(clean_accuracy, color='gray', linestyle=":", linewidth=0.8)
    
    clean_accuracy = reduce(get_score_list(results["CodeXGLUE"]["GraphCodeBERT"]["no_transformation"]["no_transformation"]))
    ax.plot(3.0, clean_accuracy, marker="*", color="gray", markersize=6.2)
    ax.axhline(clean_accuracy, color='gray', linestyle=":", linewidth=0.8)
    
    clean_accuracy = reduce(get_score_list(results["CodeXGLUE"]["CodeBERT"]["no_transformation"]["no_transformation"]))
    ax.plot(4.0, clean_accuracy, marker="*", color="gray", markersize=6.2)
    ax.axhline(clean_accuracy, color='gray', linestyle=":", linewidth=0.8)

    clean_accuracy = reduce(get_score_list(results["CodeXGLUE"]["VulBERTa"]["no_transformation"]["no_transformation"]))
    ax.plot(5.0, clean_accuracy, marker="*", color="gray", markersize=6.2)
    ax.axhline(clean_accuracy, color='gray', linestyle=":", linewidth=0.8)
    
    clean_accuracy = reduce(get_score_list(results["CodeXGLUE"]["PLBart"]["no_transformation"]["no_transformation"]))
    ax.plot(6.0, clean_accuracy, marker="*", color="gray", markersize=6.2)
    ax.axhline(clean_accuracy, color='gray', linestyle=":", linewidth=0.8)
    

    ax.set_yticks(np.arange(0,1,0.01))

    #ax.set_xlim([0, 11])
    ax.set_ylim([0.55, 0.715])

    ax.set_ylabel("test set accuracy")

    handles = []
    labels = []

    bp = ax.boxplot(x=[train_no_trafo_test_trafo_UniXcoder, train_trafo_test_trafo_UniXcoder, train_no_trafo_test_trafo_cotext, train_trafo_test_trafo_cotext, train_no_trafo_test_trafo_GraphCodeBERT, train_trafo_test_trafo_GraphCodeBERT, train_no_trafo_test_trafo_CodeBERT, train_trafo_test_trafo_CodeBERT, train_no_trafo_test_trafo_vulberta, train_trafo_test_trafo_vulberta, train_no_trafo_test_trafo_plbart, train_trafo_test_trafo_plbart],  # sequence of arrays
    positions=[0.85, 1.15, 1.85, 2.15, 2.85, 3.15, 3.85, 4.15, 4.85, 5.15, 5.85, 6.15],   # where to put these arrays
    widths=(0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13),
    patch_artist=True, sym="")
    
    for median in bp['medians']:
        median.set_color(MEDIAN_LINE_COLOR)

    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.set_xticklabels(["UniXcoder", "CoTexT", "GraphCB", "CodeBERT", "VulBERTa", "PLBart"],rotation=0, fontsize=X_TICK_LABELS_FONT_SIZE)

    patch_colors = [COLORS[0], COLORS[1], COLORS[0], COLORS[1], COLORS[0], COLORS[1], COLORS[0], COLORS[1], COLORS[0], COLORS[1], COLORS[0], COLORS[1]]
    for patch, color in zip(bp['boxes'], patch_colors):
            patch.set_facecolor(color)

            
    from matplotlib.lines import Line2D
    handles.append(Line2D([0], [0], color='gray', linestyle=":", marker="*", markersize=6.2, linewidth=0.8))
    labels.append("Train: $Tr$, Test: $Te$")
    handles.append(Line2D([0], [0], color=COLORS[0], lw=0.0, marker="s", markersize=8))
    labels.append("Train: $Tr$, Test: $Te_k$")
    handles.append(Line2D([0], [0], color=COLORS[1], lw=0.0, marker="s", markersize=8))
    labels.append("Train: $Tr_k$, Test: $Te_k$")

    legend = ax.legend(handles=handles, labels=labels,loc='upper right', frameon=True)
    legend.get_frame().set_alpha(1)

    fig.tight_layout(pad=0.02)

    figname = inspect.stack()[0][3]
    plt.savefig("{}{}.pdf".format(OUT_DIR, figname), dpi=DPI, format="pdf")
    
    
def fig_5_a():
    
    results = load_results()

    x = np.arange(1,11,1)

    fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH, FIG_HEIGHT))

    ax.set_xticks(x)
    ax.set_yticks(np.arange(0,1,0.02))

    #ax.set_xlim([0, 11])
    ax.set_ylim([0.465, 0.69])

    ax.set_xlabel("training epoch")
    ax.set_ylabel("test set accuracy")

    #ax.yaxis.grid(color='gray')
    
    handles = []
    labels = []
    
    for y in np.arange(0.48, 0.7, 0.02):
        ax.axhline(y = y, color = 'lightgray', linestyle = '-', lw=0.5)


    p = ax.plot(x, get_score_list(results["CodeXGLUE"]["VulBERTa"]["no_transformation"]["no_transformation"]), color = "gray", marker="o", markersize=4.8, linestyle=":")
    p = ax.plot(x, get_score_list(results["CodeXGLUE"]["VulBERTa"]["no_transformation"]["tf_10"]), marker='o', color = COLORS[0], markersize=4.8, linestyle="-.")
    p = ax.plot(x, get_score_list(results["CodeXGLUE"]["VulBERTa"]["tf_10"]["tf_10"]), marker='o', color = COLORS[1], markersize=4.8)
#    p = ax.plot(x, get_score_list(results["CodeXGLUE"]["VulBERTa"]["RG"]["no_transformation"]), marker='o', color = "#000000", linestyle="--", markersize=4.8)
            
    
    from matplotlib.lines import Line2D
    handles.append(Line2D([0], [0], color='gray', linestyle=":"))
    labels.append("Train: $Tr$, Test: $Te$")
    handles.append(Line2D([0], [0], color = COLORS[0], linestyle="-."))
    labels.append("Train: $Tr$, Test: $Te_{10}$")
    handles.append(Line2D([0], [0], color = COLORS[1]))
    labels.append("Train: $Tr_{10}$, Test: $Te_{10}$")
    handles.append(Line2D([0], [0], color = "#000000", linestyle="--"))
    labels.append("Random Guessing")

    legend = ax.legend(ncol=2, handles=handles, labels=labels,loc='upper left', frameon=True)

    fig.tight_layout(pad=0.02)

    figname = inspect.stack()[0][3]
    plt.savefig("{}{}.pdf".format(OUT_DIR, figname), dpi=DPI, format="pdf")
    
def fig_7_b():
    
    METRIC_TO_YLIM = {
        'accuracy' : [0.55, 0.715],
        "f1" : [0.50, 0.69], 
        "recall" : [0.3, 1.05], 
        "precision" : [0.42, 1.01], 
        "FPR" : [0.1, 1.01], 
        "FNR" : [0.1, 1.01]
    }
    
    METRIC_TO_XTICKS = {
        'accuracy' : np.arange(0,1,0.01),
        "f1" : np.arange(0,1,0.03), 
        "recall" : np.arange(0,1.1,0.05), 
        "precision" : np.arange(0,1.1,0.05), 
        "FPR" : np.arange(0,1.1,0.05), 
        "FNR" : np.arange(0,1.1,0.05)
    }
    
    METRIC_TO_LEGEND = {
        'accuracy' : True,
        "f1" : False, 
        "recall" : False, 
        "precision" : False, 
        "FPR" : False, 
        "FNR" : False
    }
    
    
    results = load_results()
    
    metric = "accuracy"
    
    train_no_trafo_test_trafo_vulberta = []
    train_trafo_test_trafo_vulberta = []
    train_other_trafo_test_trafo_vulberta = []
    train_no_trafo_test_trafo_cotext = []
    train_trafo_test_trafo_cotext = []
    train_other_trafo_test_trafo_cotext = []
    train_no_trafo_test_trafo_plbart = []
    train_trafo_test_trafo_plbart = []
    train_other_trafo_test_trafo_plbart = []
    train_no_trafo_test_trafo_UniXcoder = []
    train_trafo_test_trafo_UniXcoder = []
    train_other_trafo_test_trafo_UniXcoder = []
    train_no_trafo_test_trafo_CodeBERT = []
    train_trafo_test_trafo_CodeBERT = []
    train_other_trafo_test_trafo_CodeBERT = []
    train_no_trafo_test_trafo_GraphCodeBERT = []
    train_trafo_test_trafo_GraphCodeBERT = []
    train_other_trafo_test_trafo_GraphCodeBERT = []
    
    for transformation in SEMANTIC_PRESERVING_TRANSFORMATIONS:
        if transformation != "no_transformation":
            train_no_trafo_test_trafo_vulberta.append(reduce(get_score_list(results["CodeXGLUE"]["VulBERTa"]["no_transformation"][transformation], metric = metric)))
            train_no_trafo_test_trafo_cotext.append(reduce(get_score_list(results["CodeXGLUE"]["CoTexT"]["no_transformation"][transformation], metric = metric)))
            train_no_trafo_test_trafo_plbart.append(reduce(get_score_list(results["CodeXGLUE"]["PLBart"]["no_transformation"][transformation], metric = metric)))
            train_no_trafo_test_trafo_UniXcoder.append(reduce(get_score_list(results["CodeXGLUE"]["UniXcoder"]["no_transformation"][transformation], metric = metric)))
            train_no_trafo_test_trafo_CodeBERT.append(reduce(get_score_list(results["CodeXGLUE"]["CodeBERT"]["no_transformation"][transformation], metric = metric)))
            train_no_trafo_test_trafo_GraphCodeBERT.append(reduce(get_score_list(results["CodeXGLUE"]["GraphCodeBERT"]["no_transformation"][transformation], metric = metric)))
            
            train_trafo_test_trafo_cotext.append(reduce(get_score_list(results["CodeXGLUE"]["CoTexT"][transformation][transformation], metric = metric)))
            train_trafo_test_trafo_vulberta.append(reduce(get_score_list(results["CodeXGLUE"]["VulBERTa"][transformation][transformation], metric = metric)))
            
            if transformation in results["CodeXGLUE"]["PLBart"].keys():
                train_trafo_test_trafo_plbart.append(reduce(get_score_list(results["CodeXGLUE"]["PLBart"][transformation][transformation], metric = metric)))
                
            if transformation in results["CodeXGLUE"]["UniXcoder"].keys():
                train_trafo_test_trafo_UniXcoder.append(reduce(get_score_list(results["CodeXGLUE"]["UniXcoder"][transformation][transformation], metric = metric)))
                
            if transformation in results["CodeXGLUE"]["CodeBERT"].keys():
                train_trafo_test_trafo_CodeBERT.append(reduce(get_score_list(results["CodeXGLUE"]["CodeBERT"][transformation][transformation], metric = metric)))
                
            if transformation in results["CodeXGLUE"]["GraphCodeBERT"].keys():
                train_trafo_test_trafo_GraphCodeBERT.append(reduce(get_score_list(results["CodeXGLUE"]["GraphCodeBERT"][transformation][transformation], metric = metric)))
                
            for other_trafo in SEMANTIC_PRESERVING_TRANSFORMATIONS:
                if transformation != other_trafo and other_trafo != "tf_11":
                    train_other_trafo_test_trafo_cotext.append(reduce(get_score_list(results["CodeXGLUE"]["CoTexT"][other_trafo][transformation], metric = metric)))
                    train_other_trafo_test_trafo_vulberta.append(reduce(get_score_list(results["CodeXGLUE"]["VulBERTa"][other_trafo][transformation], metric = metric)))
                    
                    if other_trafo in results["CodeXGLUE"]["PLBart"].keys():
                        train_other_trafo_test_trafo_plbart.append(reduce(get_score_list(results["CodeXGLUE"]["PLBart"][other_trafo][transformation], metric = metric)))
                        
                    if other_trafo in results["CodeXGLUE"]["UniXcoder"].keys():
                        train_other_trafo_test_trafo_UniXcoder.append(reduce(get_score_list(results["CodeXGLUE"]["UniXcoder"][other_trafo][transformation], metric = metric)))
                        
                    if other_trafo in results["CodeXGLUE"]["CodeBERT"].keys():
                        train_other_trafo_test_trafo_CodeBERT.append(reduce(get_score_list(results["CodeXGLUE"]["CodeBERT"][other_trafo][transformation], metric = metric)))
                        
                    if other_trafo in results["CodeXGLUE"]["GraphCodeBERT"].keys():
                        train_other_trafo_test_trafo_GraphCodeBERT.append(reduce(get_score_list(results["CodeXGLUE"]["GraphCodeBERT"][other_trafo][transformation], metric = metric)))


    fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH, FIG_HEIGHT))
    
    clean_accuracy = reduce(get_score_list(results["CodeXGLUE"]["UniXcoder"]["no_transformation"]["no_transformation"], metric = metric))
    ax.plot(0.85, clean_accuracy, marker="*", color="gray", markersize=6.2)
    ax.axhline(clean_accuracy, color='gray', linestyle=":", linewidth=0.8)

    clean_accuracy = reduce(get_score_list(results["CodeXGLUE"]["CoTexT"]["no_transformation"]["no_transformation"], metric = metric))
    ax.plot(1.85, clean_accuracy, marker="*", color="gray", markersize=6.2)
    ax.axhline(clean_accuracy, color='gray', linestyle=":", linewidth=0.8)
    
    clean_accuracy = reduce(get_score_list(results["CodeXGLUE"]["GraphCodeBERT"]["no_transformation"]["no_transformation"], metric = metric))
    ax.plot(2.85, clean_accuracy, marker="*", color="gray", markersize=6.2)
    ax.axhline(clean_accuracy, color='gray', linestyle=":", linewidth=0.8)
    
    clean_accuracy = reduce(get_score_list(results["CodeXGLUE"]["CodeBERT"]["no_transformation"]["no_transformation"], metric = metric))
    ax.plot(3.85, clean_accuracy, marker="*", color="gray", markersize=6.2)
    ax.axhline(clean_accuracy, color='gray', linestyle=":", linewidth=0.8)

    clean_accuracy = reduce(get_score_list(results["CodeXGLUE"]["VulBERTa"]["no_transformation"]["no_transformation"], metric = metric))
    ax.plot(4.85, clean_accuracy, marker="*", color="gray", markersize=6.2)
    ax.axhline(clean_accuracy, color='gray', linestyle=":", linewidth=0.8)
    
    clean_accuracy = reduce(get_score_list(results["CodeXGLUE"]["PLBart"]["no_transformation"]["no_transformation"], metric = metric))
    ax.plot(5.85, clean_accuracy, marker="*", color="gray", markersize=6.2)
    ax.axhline(clean_accuracy, color='gray', linestyle=":", linewidth=0.8)
    

    ax.set_yticks(METRIC_TO_XTICKS[metric])

    #ax.set_xlim([0, 11])
    ax.set_ylim(METRIC_TO_YLIM[metric])

    ax.set_ylabel("test set {}".format(metric))

    handles = []
    labels = []

    bp = ax.boxplot(x=[train_no_trafo_test_trafo_UniXcoder, train_trafo_test_trafo_UniXcoder, train_other_trafo_test_trafo_UniXcoder, train_no_trafo_test_trafo_cotext, train_trafo_test_trafo_cotext, train_other_trafo_test_trafo_cotext, train_no_trafo_test_trafo_GraphCodeBERT, train_trafo_test_trafo_GraphCodeBERT, train_other_trafo_test_trafo_GraphCodeBERT, train_no_trafo_test_trafo_CodeBERT, train_trafo_test_trafo_CodeBERT, train_other_trafo_test_trafo_CodeBERT, train_no_trafo_test_trafo_vulberta, train_trafo_test_trafo_vulberta, train_other_trafo_test_trafo_vulberta, train_no_trafo_test_trafo_plbart, train_trafo_test_trafo_plbart, train_other_trafo_test_trafo_plbart],  # sequence of arrays
    positions=[0.7, 1.0, 1.2, 1.7, 2.0, 2.2, 2.7, 3.0, 3.2, 3.7, 4.0, 4.2, 4.7, 5.0, 5.2, 5.7, 6.0, 6.2],   # where to put these arrays
    widths=(0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13),
    patch_artist=True, sym="")
    
    for median in bp['medians']:
        median.set_color(MEDIAN_LINE_COLOR)

    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.set_xticklabels(["UniXcoder", "CoTexT", "GraphCB", "CodeBERT", "VulBERTa", "PLBart"],rotation=0, fontsize=X_TICK_LABELS_FONT_SIZE)

    patch_colors = [COLORS[0], COLORS[1], COLORS[2],COLORS[0], COLORS[1], COLORS[2],COLORS[0], COLORS[1], COLORS[2],COLORS[0], COLORS[1], COLORS[2],COLORS[0], COLORS[1], COLORS[2],COLORS[0], COLORS[1], COLORS[2]]
    for patch, color in zip(bp['boxes'], patch_colors):
            patch.set_facecolor(color)

            
    from matplotlib.lines import Line2D
    handles.append(Line2D([0], [0], color='gray', linestyle=":", marker="*", markersize=6.2, linewidth=0.8))
    labels.append("Train: $Tr$, Test: $Te$")
    handles.append(Line2D([0], [0], color=COLORS[0], lw=0.0, marker="s", markersize=8))
    labels.append("Train: $Tr$, Test: $Te_k$")
    handles.append(Line2D([0], [0], color=COLORS[1], lw=0.0, marker="s", markersize=8))
    labels.append("Train: $Tr_k$, Test: $Te_k$")
    handles.append(Line2D([0], [0], color=COLORS[2], lw=0.0, marker="s", markersize=8))
    labels.append("Train: $Tr_k$, Test: $Te_{j \\neq k}$")

    if (METRIC_TO_LEGEND[metric]):
        legend = ax.legend(handles=handles, labels=labels,loc='upper right', frameon=True)
        legend.get_frame().set_alpha(1)

    fig.tight_layout(pad=0.02)

    figname = inspect.stack()[0][3]
    plt.savefig("{}{}.pdf".format(OUT_DIR, figname), dpi=DPI, format="pdf")
        
def fig_10():
    
    METRIC_TO_YLIM = {
        'accuracy' : [0.55, 0.715],
        "f1" : [0.50, 0.69], 
        "recall" : [0.3, 1.05], 
        "precision" : [0.42, 1.01], 
        "FPR" : [0.1, 1.01], 
        "FNR" : [0.1, 1.01]
    }
    
    METRIC_TO_XTICKS = {
        'accuracy' : np.arange(0,1,0.01),
        "f1" : np.arange(0,1,0.03), 
        "recall" : np.arange(0,1.1,0.05), 
        "precision" : np.arange(0,1.1,0.05), 
        "FPR" : np.arange(0,1.1,0.05), 
        "FNR" : np.arange(0,1.1,0.05)
    }
    
    METRIC_TO_LEGEND = {
        'accuracy' : True,
        "f1" : False, 
        "recall" : False, 
        "precision" : False, 
        "FPR" : False, 
        "FNR" : False
    }
    
    
    results = load_results()
    
    for metric in METRICS:
    
        train_no_trafo_test_trafo_vulberta = []
        train_trafo_test_trafo_vulberta = []
        train_other_trafo_test_trafo_vulberta = []
        train_no_trafo_test_trafo_cotext = []
        train_trafo_test_trafo_cotext = []
        train_other_trafo_test_trafo_cotext = []
        train_no_trafo_test_trafo_plbart = []
        train_trafo_test_trafo_plbart = []
        train_other_trafo_test_trafo_plbart = []
        train_no_trafo_test_trafo_UniXcoder = []
        train_trafo_test_trafo_UniXcoder = []
        train_other_trafo_test_trafo_UniXcoder = []
        train_no_trafo_test_trafo_CodeBERT = []
        train_trafo_test_trafo_CodeBERT = []
        train_other_trafo_test_trafo_CodeBERT = []
        train_no_trafo_test_trafo_GraphCodeBERT = []
        train_trafo_test_trafo_GraphCodeBERT = []
        train_other_trafo_test_trafo_GraphCodeBERT = []
        
        for transformation in SEMANTIC_PRESERVING_TRANSFORMATIONS:
            if transformation != "no_transformation":
                train_no_trafo_test_trafo_vulberta.append(reduce(get_score_list(results["CodeXGLUE"]["VulBERTa"]["no_transformation"][transformation], metric = metric)))
                train_no_trafo_test_trafo_cotext.append(reduce(get_score_list(results["CodeXGLUE"]["CoTexT"]["no_transformation"][transformation], metric = metric)))
                train_no_trafo_test_trafo_plbart.append(reduce(get_score_list(results["CodeXGLUE"]["PLBart"]["no_transformation"][transformation], metric = metric)))
                train_no_trafo_test_trafo_UniXcoder.append(reduce(get_score_list(results["CodeXGLUE"]["UniXcoder"]["no_transformation"][transformation], metric = metric)))
                train_no_trafo_test_trafo_CodeBERT.append(reduce(get_score_list(results["CodeXGLUE"]["CodeBERT"]["no_transformation"][transformation], metric = metric)))
                train_no_trafo_test_trafo_GraphCodeBERT.append(reduce(get_score_list(results["CodeXGLUE"]["GraphCodeBERT"]["no_transformation"][transformation], metric = metric)))
                
                train_trafo_test_trafo_cotext.append(reduce(get_score_list(results["CodeXGLUE"]["CoTexT"][transformation][transformation], metric = metric)))
                train_trafo_test_trafo_vulberta.append(reduce(get_score_list(results["CodeXGLUE"]["VulBERTa"][transformation][transformation], metric = metric)))
                
                if transformation in results["CodeXGLUE"]["PLBart"].keys():
                    train_trafo_test_trafo_plbart.append(reduce(get_score_list(results["CodeXGLUE"]["PLBart"][transformation][transformation], metric = metric)))
                    
                if transformation in results["CodeXGLUE"]["UniXcoder"].keys():
                    train_trafo_test_trafo_UniXcoder.append(reduce(get_score_list(results["CodeXGLUE"]["UniXcoder"][transformation][transformation], metric = metric)))
                    
                if transformation in results["CodeXGLUE"]["CodeBERT"].keys():
                    train_trafo_test_trafo_CodeBERT.append(reduce(get_score_list(results["CodeXGLUE"]["CodeBERT"][transformation][transformation], metric = metric)))
                    
                if transformation in results["CodeXGLUE"]["GraphCodeBERT"].keys():
                    train_trafo_test_trafo_GraphCodeBERT.append(reduce(get_score_list(results["CodeXGLUE"]["GraphCodeBERT"][transformation][transformation], metric = metric)))
                    
                for other_trafo in SEMANTIC_PRESERVING_TRANSFORMATIONS:
                    if transformation != other_trafo and other_trafo != "tf_11":
                        train_other_trafo_test_trafo_cotext.append(reduce(get_score_list(results["CodeXGLUE"]["CoTexT"][other_trafo][transformation], metric = metric)))
                        train_other_trafo_test_trafo_vulberta.append(reduce(get_score_list(results["CodeXGLUE"]["VulBERTa"][other_trafo][transformation], metric = metric)))
                        
                        if other_trafo in results["CodeXGLUE"]["PLBart"].keys():
                            train_other_trafo_test_trafo_plbart.append(reduce(get_score_list(results["CodeXGLUE"]["PLBart"][other_trafo][transformation], metric = metric)))
                            
                        if other_trafo in results["CodeXGLUE"]["UniXcoder"].keys():
                            train_other_trafo_test_trafo_UniXcoder.append(reduce(get_score_list(results["CodeXGLUE"]["UniXcoder"][other_trafo][transformation], metric = metric)))
                            
                        if other_trafo in results["CodeXGLUE"]["CodeBERT"].keys():
                            train_other_trafo_test_trafo_CodeBERT.append(reduce(get_score_list(results["CodeXGLUE"]["CodeBERT"][other_trafo][transformation], metric = metric)))
                            
                        if other_trafo in results["CodeXGLUE"]["GraphCodeBERT"].keys():
                            train_other_trafo_test_trafo_GraphCodeBERT.append(reduce(get_score_list(results["CodeXGLUE"]["GraphCodeBERT"][other_trafo][transformation], metric = metric)))


        fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH, FIG_HEIGHT))
        
        clean_accuracy = reduce(get_score_list(results["CodeXGLUE"]["UniXcoder"]["no_transformation"]["no_transformation"], metric = metric))
        ax.plot(0.85, clean_accuracy, marker="*", color="gray", markersize=6.2)
        ax.axhline(clean_accuracy, color='gray', linestyle=":", linewidth=0.8)

        clean_accuracy = reduce(get_score_list(results["CodeXGLUE"]["CoTexT"]["no_transformation"]["no_transformation"], metric = metric))
        ax.plot(1.85, clean_accuracy, marker="*", color="gray", markersize=6.2)
        ax.axhline(clean_accuracy, color='gray', linestyle=":", linewidth=0.8)
        
        clean_accuracy = reduce(get_score_list(results["CodeXGLUE"]["GraphCodeBERT"]["no_transformation"]["no_transformation"], metric = metric))
        ax.plot(2.85, clean_accuracy, marker="*", color="gray", markersize=6.2)
        ax.axhline(clean_accuracy, color='gray', linestyle=":", linewidth=0.8)
        
        clean_accuracy = reduce(get_score_list(results["CodeXGLUE"]["CodeBERT"]["no_transformation"]["no_transformation"], metric = metric))
        ax.plot(3.85, clean_accuracy, marker="*", color="gray", markersize=6.2)
        ax.axhline(clean_accuracy, color='gray', linestyle=":", linewidth=0.8)

        clean_accuracy = reduce(get_score_list(results["CodeXGLUE"]["VulBERTa"]["no_transformation"]["no_transformation"], metric = metric))
        ax.plot(4.85, clean_accuracy, marker="*", color="gray", markersize=6.2)
        ax.axhline(clean_accuracy, color='gray', linestyle=":", linewidth=0.8)
        
        clean_accuracy = reduce(get_score_list(results["CodeXGLUE"]["PLBart"]["no_transformation"]["no_transformation"], metric = metric))
        ax.plot(5.85, clean_accuracy, marker="*", color="gray", markersize=6.2)
        ax.axhline(clean_accuracy, color='gray', linestyle=":", linewidth=0.8)
        

        ax.set_yticks(METRIC_TO_XTICKS[metric])

        #ax.set_xlim([0, 11])
        ax.set_ylim(METRIC_TO_YLIM[metric])

        ax.set_ylabel("test set {}".format(metric))

        handles = []
        labels = []

        bp = ax.boxplot(x=[train_no_trafo_test_trafo_UniXcoder, train_trafo_test_trafo_UniXcoder, train_other_trafo_test_trafo_UniXcoder, train_no_trafo_test_trafo_cotext, train_trafo_test_trafo_cotext, train_other_trafo_test_trafo_cotext, train_no_trafo_test_trafo_GraphCodeBERT, train_trafo_test_trafo_GraphCodeBERT, train_other_trafo_test_trafo_GraphCodeBERT, train_no_trafo_test_trafo_CodeBERT, train_trafo_test_trafo_CodeBERT, train_other_trafo_test_trafo_CodeBERT, train_no_trafo_test_trafo_vulberta, train_trafo_test_trafo_vulberta, train_other_trafo_test_trafo_vulberta, train_no_trafo_test_trafo_plbart, train_trafo_test_trafo_plbart, train_other_trafo_test_trafo_plbart],  # sequence of arrays
        positions=[0.7, 1.0, 1.2, 1.7, 2.0, 2.2, 2.7, 3.0, 3.2, 3.7, 4.0, 4.2, 4.7, 5.0, 5.2, 5.7, 6.0, 6.2],   # where to put these arrays
        widths=(0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13),
        patch_artist=True, sym="")
        
        for median in bp['medians']:
            median.set_color(MEDIAN_LINE_COLOR)

        ax.set_xticks([1, 2, 3, 4, 5, 6])
        ax.set_xticklabels(["UniXcoder", "CoTexT", "GraphCB", "CodeBERT", "VulBERTa", "PLBart"],rotation=0, fontsize=X_TICK_LABELS_FONT_SIZE)

        patch_colors = [COLORS[0], COLORS[1], COLORS[2],COLORS[0], COLORS[1], COLORS[2],COLORS[0], COLORS[1], COLORS[2],COLORS[0], COLORS[1], COLORS[2],COLORS[0], COLORS[1], COLORS[2],COLORS[0], COLORS[1], COLORS[2]]
        for patch, color in zip(bp['boxes'], patch_colors):
                patch.set_facecolor(color)

                
        from matplotlib.lines import Line2D
        handles.append(Line2D([0], [0], color='gray', linestyle=":", marker="*", markersize=6.2, linewidth=0.8))
        labels.append("Train: $Tr$, Test: $Te$")
        handles.append(Line2D([0], [0], color=COLORS[0], lw=0.0, marker="s", markersize=8))
        labels.append("Train: $Tr$, Test: $Te_k$")
        handles.append(Line2D([0], [0], color=COLORS[1], lw=0.0, marker="s", markersize=8))
        labels.append("Train: $Tr_k$, Test: $Te_k$")
        handles.append(Line2D([0], [0], color=COLORS[2], lw=0.0, marker="s", markersize=8))
        labels.append("Train: $Tr_k$, Test: $Te_{j \\neq k}$")

        if (METRIC_TO_LEGEND[metric]):
            legend = ax.legend(handles=handles, labels=labels,loc='upper right', frameon=True)
            legend.get_frame().set_alpha(1)

        fig.tight_layout(pad=0.02)

        figname = inspect.stack()[0][3] + "_" + metric
        plt.savefig("{}{}.pdf".format(OUT_DIR, figname), dpi=DPI, format="pdf")
        
def fig_7_d():
    
    
    results = load_results()
    
    for metric in ["accuracy"]:
    
        train_no_trafo_test_trafo_vulberta = []
        train_trafo_test_trafo_vulberta = []
        train_other_trafo_test_trafo_vulberta = []
        VulBERTa_10v1 = []
        train_no_trafo_test_trafo_cotext = []
        train_trafo_test_trafo_cotext = []
        train_other_trafo_test_trafo_cotext = []
        CoTexT_10v1 = []
        train_no_trafo_test_trafo_plbart = []
        train_trafo_test_trafo_plbart = []
        train_other_trafo_test_trafo_plbart = []
        PLBart_10v1 = []
        train_no_trafo_test_trafo_UniXcoder = []
        train_trafo_test_trafo_UniXcoder = []
        train_other_trafo_test_trafo_UniXcoder = []
        Unixcoder_10v1 = []
        train_no_trafo_test_trafo_CodeBERT = []
        train_trafo_test_trafo_CodeBERT = []
        train_other_trafo_test_trafo_CodeBERT = []
        CodeBERT_10v1 = []
        train_no_trafo_test_trafo_GraphCodeBERT = []
        train_trafo_test_trafo_GraphCodeBERT = []
        train_other_trafo_test_trafo_GraphCodeBERT = []
        GraphCodeBERT_10v1 = []
        
        for transformation in SEMANTIC_PRESERVING_TRANSFORMATIONS:
            if transformation != "no_transformation":
                train_no_trafo_test_trafo_vulberta.append(reduce(get_score_list(results["CodeXGLUE"]["VulBERTa"]["no_transformation"][transformation], metric = metric)))
                train_no_trafo_test_trafo_cotext.append(reduce(get_score_list(results["CodeXGLUE"]["CoTexT"]["no_transformation"][transformation], metric = metric)))
                train_no_trafo_test_trafo_plbart.append(reduce(get_score_list(results["CodeXGLUE"]["PLBart"]["no_transformation"][transformation], metric = metric)))
                train_no_trafo_test_trafo_UniXcoder.append(reduce(get_score_list(results["CodeXGLUE"]["UniXcoder"]["no_transformation"][transformation], metric = metric)))
                train_no_trafo_test_trafo_CodeBERT.append(reduce(get_score_list(results["CodeXGLUE"]["CodeBERT"]["no_transformation"][transformation], metric = metric)))
                train_no_trafo_test_trafo_GraphCodeBERT.append(reduce(get_score_list(results["CodeXGLUE"]["GraphCodeBERT"]["no_transformation"][transformation], metric = metric)))
                
                train_trafo_test_trafo_cotext.append(reduce(get_score_list(results["CodeXGLUE"]["CoTexT"][transformation][transformation], metric = metric)))
                train_trafo_test_trafo_vulberta.append(reduce(get_score_list(results["CodeXGLUE"]["VulBERTa"][transformation][transformation], metric = metric)))
                
                if transformation in results["CodeXGLUE"]["PLBart"].keys():
                    train_trafo_test_trafo_plbart.append(reduce(get_score_list(results["CodeXGLUE"]["PLBart"][transformation][transformation], metric = metric)))
                    
                if transformation in results["CodeXGLUE"]["UniXcoder"].keys():
                    train_trafo_test_trafo_UniXcoder.append(reduce(get_score_list(results["CodeXGLUE"]["UniXcoder"][transformation][transformation], metric = metric)))
                    
                if transformation in results["CodeXGLUE"]["CodeBERT"].keys():
                    train_trafo_test_trafo_CodeBERT.append(reduce(get_score_list(results["CodeXGLUE"]["CodeBERT"][transformation][transformation], metric = metric)))
                    
                if transformation in results["CodeXGLUE"]["GraphCodeBERT"].keys():
                    train_trafo_test_trafo_GraphCodeBERT.append(reduce(get_score_list(results["CodeXGLUE"]["GraphCodeBERT"][transformation][transformation], metric = metric)))
                    
                if transformation != "tf_11":
                    CodeBERT_10v1.append(reduce(get_score_list(results["CodeXGLUE"]["CodeBERT"]["10v1"][transformation], metric = metric)))
                    Unixcoder_10v1.append(reduce(get_score_list(results["CodeXGLUE"]["UniXcoder"]["10v1"][transformation], metric = metric)))
                    CoTexT_10v1.append(reduce(get_score_list(results["CodeXGLUE"]["CoTexT"]["10v1"][transformation], metric = metric)))
                    GraphCodeBERT_10v1.append(reduce(get_score_list(results["CodeXGLUE"]["GraphCodeBERT"]["10v1"][transformation], metric = metric)))
                    VulBERTa_10v1.append(reduce(get_score_list(results["CodeXGLUE"]["VulBERTa"]["10v1"][transformation], metric = metric)))
                    PLBart_10v1.append(reduce(get_score_list(results["CodeXGLUE"]["PLBart"]["10v1"][transformation], metric = metric)))
                    
                for other_trafo in SEMANTIC_PRESERVING_TRANSFORMATIONS:
                    if transformation != other_trafo and other_trafo != "tf_11":
                        train_other_trafo_test_trafo_cotext.append(reduce(get_score_list(results["CodeXGLUE"]["CoTexT"][other_trafo][transformation], metric = metric)))
                        train_other_trafo_test_trafo_vulberta.append(reduce(get_score_list(results["CodeXGLUE"]["VulBERTa"][other_trafo][transformation], metric = metric)))
                        
                        if other_trafo in results["CodeXGLUE"]["PLBart"].keys():
                            train_other_trafo_test_trafo_plbart.append(reduce(get_score_list(results["CodeXGLUE"]["PLBart"][other_trafo][transformation], metric = metric)))
                            
                        if other_trafo in results["CodeXGLUE"]["UniXcoder"].keys():
                            train_other_trafo_test_trafo_UniXcoder.append(reduce(get_score_list(results["CodeXGLUE"]["UniXcoder"][other_trafo][transformation], metric = metric)))
                            
                        if other_trafo in results["CodeXGLUE"]["CodeBERT"].keys():
                            train_other_trafo_test_trafo_CodeBERT.append(reduce(get_score_list(results["CodeXGLUE"]["CodeBERT"][other_trafo][transformation], metric = metric)))
                            
                        if other_trafo in results["CodeXGLUE"]["GraphCodeBERT"].keys():
                            train_other_trafo_test_trafo_GraphCodeBERT.append(reduce(get_score_list(results["CodeXGLUE"]["GraphCodeBERT"][other_trafo][transformation], metric = metric)))


        fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH, FIG_HEIGHT))
        
        clean_accuracy = reduce(get_score_list(results["CodeXGLUE"]["UniXcoder"]["no_transformation"]["no_transformation"], metric = metric))
        ax.plot(0.85, clean_accuracy, marker="*", color="gray", markersize=6.2)
        ax.axhline(clean_accuracy, color='gray', linestyle=":", linewidth=0.8)

        clean_accuracy = reduce(get_score_list(results["CodeXGLUE"]["CoTexT"]["no_transformation"]["no_transformation"], metric = metric))
        ax.plot(1.85, clean_accuracy, marker="*", color="gray", markersize=6.2)
        ax.axhline(clean_accuracy, color='gray', linestyle=":", linewidth=0.8)
        
        clean_accuracy = reduce(get_score_list(results["CodeXGLUE"]["GraphCodeBERT"]["no_transformation"]["no_transformation"], metric = metric))
        ax.plot(2.85, clean_accuracy, marker="*", color="gray", markersize=6.2)
        ax.axhline(clean_accuracy, color='gray', linestyle=":", linewidth=0.8)
        
        clean_accuracy = reduce(get_score_list(results["CodeXGLUE"]["CodeBERT"]["no_transformation"]["no_transformation"], metric = metric))
        ax.plot(3.85, clean_accuracy, marker="*", color="gray", markersize=6.2)
        ax.axhline(clean_accuracy, color='gray', linestyle=":", linewidth=0.8)

        clean_accuracy = reduce(get_score_list(results["CodeXGLUE"]["VulBERTa"]["no_transformation"]["no_transformation"], metric = metric))
        ax.plot(4.85, clean_accuracy, marker="*", color="gray", markersize=6.2)
        ax.axhline(clean_accuracy, color='gray', linestyle=":", linewidth=0.8)
        
        clean_accuracy = reduce(get_score_list(results["CodeXGLUE"]["PLBart"]["no_transformation"]["no_transformation"], metric = metric))
        ax.plot(5.85, clean_accuracy, marker="*", color="gray", markersize=6.2)
        ax.axhline(clean_accuracy, color='gray', linestyle=":", linewidth=0.8)
        

        ax.set_yticks(np.arange(0,1,0.01))

        #ax.set_xlim([0, 11])
        ax.set_ylim([0.55, 0.715])

        ax.set_ylabel("test set accuracy")

        handles = []
        labels = []

        bp = ax.boxplot(x=[train_no_trafo_test_trafo_UniXcoder, train_trafo_test_trafo_UniXcoder, Unixcoder_10v1, train_no_trafo_test_trafo_cotext, train_trafo_test_trafo_cotext, CoTexT_10v1, train_no_trafo_test_trafo_GraphCodeBERT, train_trafo_test_trafo_GraphCodeBERT, GraphCodeBERT_10v1, train_no_trafo_test_trafo_CodeBERT, train_trafo_test_trafo_CodeBERT, CodeBERT_10v1, train_no_trafo_test_trafo_vulberta, train_trafo_test_trafo_vulberta, VulBERTa_10v1, train_no_trafo_test_trafo_plbart, train_trafo_test_trafo_plbart, PLBart_10v1],  # sequence of arrays
        positions=[0.7, 1.0, 1.2, 1.7, 2.0, 2.2, 2.7, 3.0, 3.2, 3.7, 4.0, 4.2, 4.7, 5.0, 5.2, 5.7, 6.0, 6.2],   # where to put these arrays
        widths=(0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13),
        patch_artist=True, sym="")
        
        for median in bp['medians']:
            median.set_color(MEDIAN_LINE_COLOR)

        ax.set_xticks([1, 2, 3, 4, 5, 6])
        ax.set_xticklabels(["UniXcoder", "CoTexT", "GraphCB", "CodeBERT", "VulBERTa", "PLBart"],rotation=0, fontsize=X_TICK_LABELS_FONT_SIZE)

        patch_colors = [COLORS[0], COLORS[1], COLORS[2],COLORS[0], COLORS[1], COLORS[2],COLORS[0], COLORS[1], COLORS[2],COLORS[0], COLORS[1], COLORS[2],COLORS[0], COLORS[1], COLORS[2],COLORS[0], COLORS[1], COLORS[2]]
        for patch, color in zip(bp['boxes'], patch_colors):
                patch.set_facecolor(color)

                
        from matplotlib.lines import Line2D
        handles.append(Line2D([0], [0], color='gray', linestyle=":", marker="*", markersize=6.2, linewidth=0.8))
        labels.append("Train: $Tr$, Test: $Te$")
        handles.append(Line2D([0], [0], color=COLORS[0], lw=0.0, marker="s", markersize=8))
        labels.append("Train: $Tr$, Test: $Te_k$")
        handles.append(Line2D([0], [0], color=COLORS[1], lw=0.0, marker="s", markersize=8))
        labels.append("Train: $Tr_k$, Test: $Te_k$")
        handles.append(Line2D([0], [0], color=COLORS[2], lw=0.0, marker="s", markersize=8))
        labels.append("Train: $Tr_{11}$, Test: $Te_j$")

        legend = ax.legend(handles=handles, labels=labels,loc='upper right', frameon=True)
        legend.get_frame().set_alpha(1)

        fig.tight_layout(pad=0.02)

        figname = inspect.stack()[0][3] + "_" + metric
        plt.savefig("{}{}.pdf".format(OUT_DIR, figname), dpi=DPI, format="pdf")
    
def fig_9():
    
    results = load_results()
    
    trafo_impact = dict()
    
    trafo_impact = np.zeros((len(TECHNIQUES), len(SEMANTIC_PRESERVING_TRANSFORMATIONS[1:])))
    for i, technique in enumerate(TECHNIQUES):
        for j, transformation in enumerate(SEMANTIC_PRESERVING_TRANSFORMATIONS[1:]):
            for k, other_trafo in enumerate(SEMANTIC_PRESERVING_TRANSFORMATIONS[1:]):
                if k != j:
                    impact = reduce(get_score_list(results["CodeXGLUE"][technique][transformation]["no_transformation"])) - reduce(get_score_list(results["CodeXGLUE"][technique][transformation][other_trafo]))
                    trafo_impact[i, k] -= impact
                
    
    trafo_impact/= len(SEMANTIC_PRESERVING_TRANSFORMATIONS[1:])


    fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH * 2, FIG_HEIGHT / 1.7))
  
    X_axis = np.arange(len(SEMANTIC_PRESERVING_TRANSFORMATIONS[1:])) 
    
    ax.axhline(y = -0.02, color = 'lightgray', linestyle = '-', lw=0.5, zorder=1)
    ax.axhline(y = -0.04, color = 'lightgray', linestyle = '-', lw=0.5, zorder=1)
    ax.axhline(y = -0.06, color = 'lightgray', linestyle = '-', lw=0.5, zorder=1)
    ax.axhline(y = -0.08, color = 'lightgray', linestyle = '-', lw=0.5, zorder=1)
    ax.axhline(y = -0.10, color = 'lightgray', linestyle = '-', lw=0.5, zorder=1)
    ax.axhline(y = -0.12, color = 'lightgray', linestyle = '-', lw=0.5, zorder=1)
    ax.axhline(y = -0.14, color = 'lightgray', linestyle = '-', lw=0.5, zorder=1)
    ax.axhline(y = -0.16, color = 'lightgray', linestyle = '-', lw=0.5, zorder=1)
    ax.axhline(y = -0.18, color = 'lightgray', linestyle = '-', lw=0.5, zorder=1)
    
    ax.bar(X_axis - 0.30, trafo_impact[0], 0.12, label = TECHNIQUES[0], zorder=2, color="black", edgecolor='black')
    ax.bar(X_axis - 0.18, trafo_impact[1], 0.12, label = TECHNIQUES[1], zorder=2, fill=False, hatch='...')
    ax.bar(X_axis - 0.06, trafo_impact[2], 0.12, label = TECHNIQUES[2], zorder=2, color="lightgray", edgecolor='black')
    ax.bar(X_axis + 0.06, trafo_impact[3], 0.12, label = TECHNIQUES[3], zorder=2, fill=False, hatch='///')
    ax.bar(X_axis + 0.18, trafo_impact[4], 0.12, label = TECHNIQUES[4], zorder=2, color="gray", edgecolor='black')
    ax.bar(X_axis + 0.30, trafo_impact[5], 0.12, label = TECHNIQUES[5], zorder=2, fill=False, hatch='xxx')
    
    ax.scatter(np.argmin(trafo_impact[0]) - 0.30 , np.min(trafo_impact[0]) - 0.007 , marker="*", s=50, color="red")
    ax.scatter(np.argmin(trafo_impact[1]) - 0.18 , np.min(trafo_impact[1]) - 0.007 , marker="*", s=50, color="red")
    ax.scatter(np.argmin(trafo_impact[2]) - 0.06 , np.min(trafo_impact[2]) - 0.007 , marker="*", s=50, color="red")
    ax.scatter(np.argmin(trafo_impact[3]) + 0.06 , np.min(trafo_impact[3]) - 0.007 , marker="*", s=50, color="red")
    ax.scatter(np.argmin(trafo_impact[4]) + 0.18 , np.min(trafo_impact[4]) - 0.007 , marker="*", s=50, color="red")
    ax.scatter(np.argmin(trafo_impact[5]) + 0.30 , np.min(trafo_impact[5]) - 0.007 , marker="*", s=50, color="red")
    
    # ax.scatter(np.argmax(trafo_impact[0]) - 0.30 , np.max(trafo_impact[0]) - 0.007 , marker="x", s=45, color="red")
    # ax.scatter(np.argmax(trafo_impact[1]) - 0.18 , np.max(trafo_impact[1]) - 0.007 , marker="x", s=45, color="red")
    # ax.scatter(np.argmax(trafo_impact[2]) - 0.06 , np.max(trafo_impact[2]) - 0.007 , marker="x", s=45, color="red")
    # ax.scatter(np.argmax(trafo_impact[3]) + 0.06 , np.max(trafo_impact[3]) - 0.007 , marker="x", s=45, color="red")
    # ax.scatter(np.argmax(trafo_impact[4]) + 0.18 , np.max(trafo_impact[4]) - 0.007 , marker="x", s=45, color="red")
    # ax.scatter(np.argmax(trafo_impact[5]) + 0.30 , np.max(trafo_impact[5]) - 0.007 , marker="x", s=45, color="red")
    
    ax.set_ylim([-0.185, 0.0])
    ax.set_yticks(np.arange(-0.18,0.02,0.02))
    
    plt.gca().invert_yaxis()
    
    ax.set_ylabel("impact($t_k$)")
    ax.set_xticks(X_axis, ["$t_{}$".format("{" + str(i+1) + "}") for i in range(11)])
    
    legend = ax.legend(ncols=6, loc='upper center', frameon=True)
    legend.get_frame().set_alpha(1)

    fig.tight_layout(pad=0.02)

    figname = inspect.stack()[0][3]
    plt.savefig("{}{}.pdf".format(OUT_DIR, figname), dpi=DPI, format="pdf")
    
def fig_6():
    
    results = load_results()
    
    trafo_impact = dict()
    
    trafo_impact = np.zeros((len(TECHNIQUES), len(SEMANTIC_PRESERVING_TRANSFORMATIONS[1:])))
    for i, technique in enumerate(TECHNIQUES):
        for j, transformation in enumerate(SEMANTIC_PRESERVING_TRANSFORMATIONS[1:]):
                impact = reduce(get_score_list(results["CodeXGLUE"][technique]["no_transformation"]["no_transformation"])) - reduce(get_score_list(results["CodeXGLUE"][technique]["no_transformation"][transformation]))
                trafo_impact[i, j] -= impact


    fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH * 2, FIG_HEIGHT / 1.7))
  
    X_axis = np.arange(len(SEMANTIC_PRESERVING_TRANSFORMATIONS[1:])) 
    
    ax.axhline(y = -0.03, color = 'lightgray', linestyle = '-', lw=0.5, zorder=1)
    ax.axhline(y = -0.06, color = 'lightgray', linestyle = '-', lw=0.5, zorder=1)
    ax.axhline(y = -0.09, color = 'lightgray', linestyle = '-', lw=0.5, zorder=1)
    ax.axhline(y = -0.12, color = 'lightgray', linestyle = '-', lw=0.5, zorder=1)
    ax.axhline(y = -0.15, color = 'lightgray', linestyle = '-', lw=0.5, zorder=1)
    ax.axhline(y = -0.18, color = 'lightgray', linestyle = '-', lw=0.5, zorder=1)
    ax.axhline(y = -0.21, color = 'lightgray', linestyle = '-', lw=0.5, zorder=1)
    ax.axhline(y = -0.24, color = 'lightgray', linestyle = '-', lw=0.5, zorder=1)
    
    ax.bar(X_axis - 0.30, trafo_impact[0], 0.12, label = TECHNIQUES[0], zorder=2, color="black", edgecolor='black')
    ax.bar(X_axis - 0.18, trafo_impact[1], 0.12, label = TECHNIQUES[1], zorder=2, fill=False, hatch='...')
    ax.bar(X_axis - 0.06, trafo_impact[2], 0.12, label = TECHNIQUES[2], zorder=2, color="lightgray", edgecolor='black')
    ax.bar(X_axis + 0.06, trafo_impact[3], 0.12, label = TECHNIQUES[3], zorder=2, fill=False, hatch='///')
    ax.bar(X_axis + 0.18, trafo_impact[4], 0.12, label = TECHNIQUES[4], zorder=2, color="gray", edgecolor='black')
    ax.bar(X_axis + 0.30, trafo_impact[5], 0.12, label = TECHNIQUES[5], zorder=2, fill=False, hatch='xxx')
    
    ax.scatter(np.argmin(trafo_impact[0]) - 0.30 , np.min(trafo_impact[0]) - 0.008 , marker="*", s=50, color="red")
    ax.scatter(np.argmin(trafo_impact[1]) - 0.18 , np.min(trafo_impact[1]) - 0.008 , marker="*", s=50, color="red")
    ax.scatter(np.argmin(trafo_impact[2]) - 0.06 , np.min(trafo_impact[2]) - 0.008 , marker="*", s=50, color="red")
    ax.scatter(np.argmin(trafo_impact[3]) + 0.06 , np.min(trafo_impact[3]) - 0.008 , marker="*", s=50, color="red")
    ax.scatter(np.argmin(trafo_impact[4]) + 0.18 , np.min(trafo_impact[4]) - 0.008 , marker="*", s=50, color="red")
    ax.scatter(np.argmin(trafo_impact[5]) + 0.30 , np.min(trafo_impact[5]) - 0.008 , marker="*", s=50, color="red")
    
    ax.set_ylim([-0.245, 0.0])
    ax.set_yticks(np.arange(-0.24,0.02,0.03))
    
    plt.gca().invert_yaxis()
    
    ax.set_ylabel("impact($t_k$)")
    ax.set_xticks(X_axis, ["$t_{}$".format("{" + str(i+1) + "}") for i in range(11)])
    
    legend = ax.legend(ncols=6, loc='upper center', frameon=True)
    legend.get_frame().set_alpha(1)

    fig.tight_layout(pad=0.02)

    figname = inspect.stack()[0][3]
    plt.savefig("{}{}.pdf".format(OUT_DIR, figname), dpi=DPI, format="pdf")

    
def fig_8():
    
    naturalness = dict()
    
    for trafo in SEMANTIC_PRESERVING_TRANSFORMATIONS:
        results_file_name = './results/naturalness_codexglue_test_{}.pkl'.format(trafo)
        if os.path.isfile(results_file_name):
                with open(results_file_name, 'rb') as handle:
                        naturalness[trafo] = pickle.load(handle)
                        

    fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH, FIG_HEIGHT))


    # clean_accuracy = reduce(df_cotext["train: tf_00 test: tf_00 - test/accuracy"].to_list())
    # ax.axhline(clean_accuracy, label="CoTexT - Train: Standard, Test: Standard", color="pink")

    # clean_accuracy = reduce(df_vulberta["train: tf_00 test: tf_00 - test/accuracy"].to_list())
    # ax.axhline(clean_accuracy, label="VulBERTa - Train: Standard, Test: Standard", color="lightblue")

    ax.set_yticks(np.arange(0,10,0.5))

    #ax.set_xlim([0, 11])
    ax.set_ylim([2, 7])

    ax.set_ylabel("cross entropy")

    handles = []
    labels = []

    boxplot_data = []
    
    for trafo in SEMANTIC_PRESERVING_TRANSFORMATIONS:
        boxplot_data.append(naturalness[trafo])
    bp = ax.boxplot(x=boxplot_data,  # sequence of arrays
    positions=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],   # where to put these arrays
    widths=(0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26),
    patch_artist=True, sym="")
    
    for median in bp['medians']:
        median.set_color(MEDIAN_LINE_COLOR)
        
    x_tick_labels = ["None", "$t_{1}$", "$t_{2}$", "$t_{3}$", "$t_{4}$", "$t_{5}$", "$t_{6}$", "$t_{7}$", "$t_{8}$", "$t_{9}$", "$t_{10}$", "$t_{11}$"]

    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    ax.set_xticklabels(x_tick_labels, rotation=0, fontsize=X_TICK_LABELS_FONT_SIZE)

    patch_colors = ["#808080", "lightgray", "lightgray", "lightgray", "lightgray", "lightgray", "lightgray", "lightgray", "lightgray", "lightgray", "lightgray", "lightgray"]
    for patch, color in zip(bp['boxes'], patch_colors):
            patch.set_facecolor(color)

            
    from matplotlib.lines import Line2D
    # handles.append(Line2D([0], [0], color="pink"))
    # labels.append("CoTexT - Train: Standard, Test: Standard")
    # handles.append(Line2D([0], [0], color="lightblue"))
    # labels.append("VulBERTa - Train: Standard, Test: Standard")
    
    clean_accuracy = np.mean(np.array(naturalness["no_transformation"]))
    ax.axhline(clean_accuracy, label="Mean cross entropy clean testing set", color="black")

    ax.legend(handles=handles, labels=labels,loc='upper left', frameon=False)
    fig.tight_layout(pad=0.02)

    figname = inspect.stack()[0][3]
    plt.savefig("{}{}.pdf".format(OUT_DIR, figname), dpi=DPI, format="pdf")

def fig_7_c():
    
    
    METRIC_TO_YLIM = {
        'accuracy' : [0.935, 1.0],
        "f1" : [0.63, 0.9], 
        "recall" : [0.4, 0.95], 
        "precision" : [0.65, 1.03], 
        "FPR" : [0, 0.15], 
        "FNR" : [0, 1]
    }
    
    METRIC_TO_XTICKS = {
        'accuracy' : np.arange(0,1.1,0.005),
        "f1" : np.arange(0,1,0.02), 
        "recall" : np.arange(0,1.1,0.05), 
        "precision" : np.arange(0,1.1,0.05), 
        "FPR" : np.arange(0,1.1,0.01), 
        "FNR" : np.arange(0,1.1,0.05)
    }
    
    METRIC_TO_LEGEND = {
        'accuracy' : True,
        "f1" : False, 
        "recall" : False, 
        "precision" : False, 
        "FPR" : False, 
        "FNR" : False
    }
    
    
    results = load_results()
    
    metric="f1"
    
    train_no_trafo_test_trafo_vulberta = []
    train_trafo_test_trafo_vulberta = []
    train_other_trafo_test_trafo_vulberta = []
    train_no_trafo_test_trafo_cotext = []
    train_trafo_test_trafo_cotext = []
    train_other_trafo_test_trafo_cotext = []
    train_no_trafo_test_trafo_plbart = []
    train_trafo_test_trafo_plbart = []
    train_other_trafo_test_trafo_plbart = []
    train_no_trafo_test_trafo_UniXcoder = []
    train_trafo_test_trafo_UniXcoder = []
    train_other_trafo_test_trafo_UniXcoder = []
    train_no_trafo_test_trafo_CodeBERT = []
    train_trafo_test_trafo_CodeBERT = []
    train_other_trafo_test_trafo_CodeBERT = []
    train_no_trafo_test_trafo_GraphCodeBERT = []
    train_trafo_test_trafo_GraphCodeBERT = []
    train_other_trafo_test_trafo_GraphCodeBERT = []
    
    for transformation in SEMANTIC_PRESERVING_TRANSFORMATIONS:
        # tf_9 (comment removal) is filtered out for VulDeePecker because we already remove comments in the pre-processing for VulDeePecker
        if transformation != "no_transformation" and transformation != "tf_9":
            train_no_trafo_test_trafo_vulberta.append(reduce(get_score_list(results["VulDeePecker"]["VulBERTa"]["no_transformation"][transformation], metric=metric)))
            train_no_trafo_test_trafo_cotext.append(reduce(get_score_list(results["VulDeePecker"]["CoTexT"]["no_transformation"][transformation], metric=metric)))
            train_no_trafo_test_trafo_plbart.append(reduce(get_score_list(results["VulDeePecker"]["PLBart"]["no_transformation"][transformation], metric=metric)))
            train_no_trafo_test_trafo_UniXcoder.append(reduce(get_score_list(results["VulDeePecker"]["UniXcoder"]["no_transformation"][transformation], metric=metric)))
            train_no_trafo_test_trafo_CodeBERT.append(reduce(get_score_list(results["VulDeePecker"]["CodeBERT"]["no_transformation"][transformation], metric=metric)))
            train_no_trafo_test_trafo_GraphCodeBERT.append(reduce(get_score_list(results["VulDeePecker"]["GraphCodeBERT"]["no_transformation"][transformation], metric=metric)))
            
            train_trafo_test_trafo_cotext.append(reduce(get_score_list(results["VulDeePecker"]["CoTexT"][transformation][transformation], metric=metric)))
            train_trafo_test_trafo_vulberta.append(reduce(get_score_list(results["VulDeePecker"]["VulBERTa"][transformation][transformation], metric=metric)))
            
            if transformation in results["VulDeePecker"]["PLBart"].keys():
                train_trafo_test_trafo_plbart.append(reduce(get_score_list(results["VulDeePecker"]["PLBart"][transformation][transformation], metric=metric)))
                
            if transformation in results["VulDeePecker"]["UniXcoder"].keys():
                train_trafo_test_trafo_UniXcoder.append(reduce(get_score_list(results["VulDeePecker"]["UniXcoder"][transformation][transformation], metric=metric)))
                
            if transformation in results["VulDeePecker"]["CodeBERT"].keys():
                train_trafo_test_trafo_CodeBERT.append(reduce(get_score_list(results["VulDeePecker"]["CodeBERT"][transformation][transformation], metric=metric)))
                
            if transformation in results["VulDeePecker"]["GraphCodeBERT"].keys():
                train_trafo_test_trafo_GraphCodeBERT.append(reduce(get_score_list(results["VulDeePecker"]["GraphCodeBERT"][transformation][transformation], metric=metric)))
                
            for other_trafo in SEMANTIC_PRESERVING_TRANSFORMATIONS:
                # tf_9 (comment removal) is filtered out for VulDeePecker because we already remove comments in the pre-processing for VulDeePecker
                if transformation != other_trafo and transformation != "tf_11" and other_trafo != "no_transformation" and other_trafo != "tf_9":
                    train_other_trafo_test_trafo_cotext.append(reduce(get_score_list(results["VulDeePecker"]["CoTexT"][other_trafo][transformation], metric=metric)))
                    train_other_trafo_test_trafo_vulberta.append(reduce(get_score_list(results["VulDeePecker"]["VulBERTa"][other_trafo][transformation], metric=metric)))
                    
                    if other_trafo in results["VulDeePecker"]["PLBart"].keys():
                        train_other_trafo_test_trafo_plbart.append(reduce(get_score_list(results["VulDeePecker"]["PLBart"][other_trafo][transformation], metric=metric)))
                        
                    if other_trafo in results["VulDeePecker"]["UniXcoder"].keys():
                        train_other_trafo_test_trafo_UniXcoder.append(reduce(get_score_list(results["VulDeePecker"]["UniXcoder"][other_trafo][transformation], metric=metric)))
                        
                    if other_trafo in results["VulDeePecker"]["CodeBERT"].keys():
                        train_other_trafo_test_trafo_CodeBERT.append(reduce(get_score_list(results["VulDeePecker"]["CodeBERT"][other_trafo][transformation], metric=metric)))
                        
                    if other_trafo in results["VulDeePecker"]["GraphCodeBERT"].keys():
                        train_other_trafo_test_trafo_GraphCodeBERT.append(reduce(get_score_list(results["VulDeePecker"]["GraphCodeBERT"][other_trafo][transformation], metric=metric)))


    fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH, FIG_HEIGHT))
    
    clean_accuracy = reduce(get_score_list(results["VulDeePecker"]["UniXcoder"]["no_transformation"]["no_transformation"], metric=metric))
    ax.plot(0.85, clean_accuracy, marker="*", color="gray", markersize=6.2)
    ax.axhline(clean_accuracy, color='gray', linestyle=":", linewidth=0.8)

    clean_accuracy = reduce(get_score_list(results["VulDeePecker"]["CoTexT"]["no_transformation"]["no_transformation"], metric=metric))
    ax.plot(1.85, clean_accuracy, marker="*", color="gray", markersize=6.2)
    ax.axhline(clean_accuracy, color='gray', linestyle=":", linewidth=0.8)
    
    clean_accuracy = reduce(get_score_list(results["VulDeePecker"]["GraphCodeBERT"]["no_transformation"]["no_transformation"], metric=metric))
    ax.plot(2.85, clean_accuracy, marker="*", color="gray", markersize=6.2)
    ax.axhline(clean_accuracy, color='gray', linestyle=":", linewidth=0.8)
    
    clean_accuracy = reduce(get_score_list(results["VulDeePecker"]["CodeBERT"]["no_transformation"]["no_transformation"], metric=metric))
    ax.plot(3.85, clean_accuracy, marker="*", color="gray", markersize=6.2)
    ax.axhline(clean_accuracy, color='gray', linestyle=":", linewidth=0.8)

    clean_accuracy = reduce(get_score_list(results["VulDeePecker"]["VulBERTa"]["no_transformation"]["no_transformation"], metric=metric))
    ax.plot(4.85, clean_accuracy, marker="*", color="gray", markersize=6.2)
    ax.axhline(clean_accuracy, color='gray', linestyle=":", linewidth=0.8)
    
    clean_accuracy = reduce(get_score_list(results["VulDeePecker"]["PLBart"]["no_transformation"]["no_transformation"], metric=metric))
    ax.plot(5.85, clean_accuracy, marker="*", color="gray", markersize=6.2)
    ax.axhline(clean_accuracy, color='gray', linestyle=":", linewidth=0.8)
    

    ax.set_yticks(METRIC_TO_XTICKS[metric])

    #ax.set_xlim([0, 11])
    ax.set_ylim(METRIC_TO_YLIM[metric])

    ax.set_ylabel("test set {}".format(metric))

    handles = []
    labels = []

    bp = ax.boxplot(x=[train_no_trafo_test_trafo_UniXcoder, train_trafo_test_trafo_UniXcoder, train_other_trafo_test_trafo_UniXcoder, train_no_trafo_test_trafo_cotext, train_trafo_test_trafo_cotext, train_other_trafo_test_trafo_cotext, train_no_trafo_test_trafo_GraphCodeBERT, train_trafo_test_trafo_GraphCodeBERT, train_other_trafo_test_trafo_GraphCodeBERT, train_no_trafo_test_trafo_CodeBERT, train_trafo_test_trafo_CodeBERT, train_other_trafo_test_trafo_CodeBERT, train_no_trafo_test_trafo_vulberta, train_trafo_test_trafo_vulberta, train_other_trafo_test_trafo_vulberta, train_no_trafo_test_trafo_plbart, train_trafo_test_trafo_plbart, train_other_trafo_test_trafo_plbart],  # sequence of arrays
    positions=[0.7, 1.0, 1.2, 1.7, 2.0, 2.2, 2.7, 3.0, 3.2, 3.7, 4.0, 4.2, 4.7, 5.0, 5.2, 5.7, 6.0, 6.2],   # where to put these arrays
    widths=(0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13),
    patch_artist=True, sym="")
    
    for median in bp['medians']:
        median.set_color(MEDIAN_LINE_COLOR)

    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.set_xticklabels(["UniXcoder", "CoTexT", "GraphCB", "CodeBERT", "VulBERTa", "PLBart"],rotation=0, fontsize=X_TICK_LABELS_FONT_SIZE)

    patch_colors = [COLORS[0], COLORS[1], COLORS[2], COLORS[0], COLORS[1], COLORS[2], COLORS[0], COLORS[1], COLORS[2], COLORS[0], COLORS[1], COLORS[2], COLORS[0], COLORS[1], COLORS[2], COLORS[0], COLORS[1], COLORS[2]]
    for patch, color in zip(bp['boxes'], patch_colors):
            patch.set_facecolor(color)

            
    from matplotlib.lines import Line2D
    handles.append(Line2D([0], [0], color='gray', linestyle=":", marker="*", markersize=6.2, linewidth=0.8))
    labels.append("Train: $Tr$, Test: $Te$")
    handles.append(Line2D([0], [0], color=COLORS[0], lw=0.0, marker="s", markersize=8))
    labels.append("Train: $Tr$, Test: $Te_k$")
    handles.append(Line2D([0], [0], color=COLORS[1], lw=0.0, marker="s", markersize=8))
    labels.append("Train: $Tr_k$, Test: $Te_k$")
    handles.append(Line2D([0], [0], color=COLORS[2], lw=0.0, marker="s", markersize=8))
    labels.append("Train: $Tr_k$, Test: $Te_{j \\neq k}$")

    if (METRIC_TO_LEGEND[metric]):
        legend = ax.legend(handles=handles, labels=labels,loc='upper right', frameon=True)
        legend.get_frame().set_alpha(1)

    fig.tight_layout(pad=0.02)

    figname = inspect.stack()[0][3]
    plt.savefig("{}{}.pdf".format(OUT_DIR, figname), dpi=DPI, format="pdf")
        
def fig_11():
    
    
    METRIC_TO_YLIM = {
        'accuracy' : [0.935, 1.0],
        "f1" : [0.63, 0.9], 
        "recall" : [0.4, 0.95], 
        "precision" : [0.65, 1.03], 
        "FPR" : [0, 0.15], 
        "FNR" : [0, 1]
    }
    
    METRIC_TO_XTICKS = {
        'accuracy' : np.arange(0,1.1,0.005),
        "f1" : np.arange(0,1,0.02), 
        "recall" : np.arange(0,1.1,0.05), 
        "precision" : np.arange(0,1.1,0.05), 
        "FPR" : np.arange(0,1.1,0.01), 
        "FNR" : np.arange(0,1.1,0.05)
    }
    
    METRIC_TO_LEGEND = {
        'accuracy' : True,
        "f1" : False, 
        "recall" : False, 
        "precision" : False, 
        "FPR" : False, 
        "FNR" : False
    }
    
    
    results = load_results()
    
    for metric in METRICS:
    
        train_no_trafo_test_trafo_vulberta = []
        train_trafo_test_trafo_vulberta = []
        train_other_trafo_test_trafo_vulberta = []
        train_no_trafo_test_trafo_cotext = []
        train_trafo_test_trafo_cotext = []
        train_other_trafo_test_trafo_cotext = []
        train_no_trafo_test_trafo_plbart = []
        train_trafo_test_trafo_plbart = []
        train_other_trafo_test_trafo_plbart = []
        train_no_trafo_test_trafo_UniXcoder = []
        train_trafo_test_trafo_UniXcoder = []
        train_other_trafo_test_trafo_UniXcoder = []
        train_no_trafo_test_trafo_CodeBERT = []
        train_trafo_test_trafo_CodeBERT = []
        train_other_trafo_test_trafo_CodeBERT = []
        train_no_trafo_test_trafo_GraphCodeBERT = []
        train_trafo_test_trafo_GraphCodeBERT = []
        train_other_trafo_test_trafo_GraphCodeBERT = []
        
        for transformation in SEMANTIC_PRESERVING_TRANSFORMATIONS:
            # tf_9 (comment removal) is filtered out for VulDeePecker because we already remove comments in the pre-processing for VulDeePecker
            if transformation != "no_transformation" and transformation != "tf_9":
                train_no_trafo_test_trafo_vulberta.append(reduce(get_score_list(results["VulDeePecker"]["VulBERTa"]["no_transformation"][transformation], metric=metric)))
                train_no_trafo_test_trafo_cotext.append(reduce(get_score_list(results["VulDeePecker"]["CoTexT"]["no_transformation"][transformation], metric=metric)))
                train_no_trafo_test_trafo_plbart.append(reduce(get_score_list(results["VulDeePecker"]["PLBart"]["no_transformation"][transformation], metric=metric)))
                train_no_trafo_test_trafo_UniXcoder.append(reduce(get_score_list(results["VulDeePecker"]["UniXcoder"]["no_transformation"][transformation], metric=metric)))
                train_no_trafo_test_trafo_CodeBERT.append(reduce(get_score_list(results["VulDeePecker"]["CodeBERT"]["no_transformation"][transformation], metric=metric)))
                train_no_trafo_test_trafo_GraphCodeBERT.append(reduce(get_score_list(results["VulDeePecker"]["GraphCodeBERT"]["no_transformation"][transformation], metric=metric)))
                
                train_trafo_test_trafo_cotext.append(reduce(get_score_list(results["VulDeePecker"]["CoTexT"][transformation][transformation], metric=metric)))
                train_trafo_test_trafo_vulberta.append(reduce(get_score_list(results["VulDeePecker"]["VulBERTa"][transformation][transformation], metric=metric)))
                
                if transformation in results["VulDeePecker"]["PLBart"].keys():
                    train_trafo_test_trafo_plbart.append(reduce(get_score_list(results["VulDeePecker"]["PLBart"][transformation][transformation], metric=metric)))
                    
                if transformation in results["VulDeePecker"]["UniXcoder"].keys():
                    train_trafo_test_trafo_UniXcoder.append(reduce(get_score_list(results["VulDeePecker"]["UniXcoder"][transformation][transformation], metric=metric)))
                    
                if transformation in results["VulDeePecker"]["CodeBERT"].keys():
                    train_trafo_test_trafo_CodeBERT.append(reduce(get_score_list(results["VulDeePecker"]["CodeBERT"][transformation][transformation], metric=metric)))
                    
                if transformation in results["VulDeePecker"]["GraphCodeBERT"].keys():
                    train_trafo_test_trafo_GraphCodeBERT.append(reduce(get_score_list(results["VulDeePecker"]["GraphCodeBERT"][transformation][transformation], metric=metric)))
                    
                for other_trafo in SEMANTIC_PRESERVING_TRANSFORMATIONS:
                    # tf_9 (comment removal) is filtered out for VulDeePecker because we already remove comments in the pre-processing for VulDeePecker
                    if transformation != other_trafo and transformation != "tf_11" and other_trafo != "no_transformation" and other_trafo != "tf_9":
                        train_other_trafo_test_trafo_cotext.append(reduce(get_score_list(results["VulDeePecker"]["CoTexT"][other_trafo][transformation], metric=metric)))
                        train_other_trafo_test_trafo_vulberta.append(reduce(get_score_list(results["VulDeePecker"]["VulBERTa"][other_trafo][transformation], metric=metric)))
                        
                        if other_trafo in results["VulDeePecker"]["PLBart"].keys():
                            train_other_trafo_test_trafo_plbart.append(reduce(get_score_list(results["VulDeePecker"]["PLBart"][other_trafo][transformation], metric=metric)))
                            
                        if other_trafo in results["VulDeePecker"]["UniXcoder"].keys():
                            train_other_trafo_test_trafo_UniXcoder.append(reduce(get_score_list(results["VulDeePecker"]["UniXcoder"][other_trafo][transformation], metric=metric)))
                            
                        if other_trafo in results["VulDeePecker"]["CodeBERT"].keys():
                            train_other_trafo_test_trafo_CodeBERT.append(reduce(get_score_list(results["VulDeePecker"]["CodeBERT"][other_trafo][transformation], metric=metric)))
                            
                        if other_trafo in results["VulDeePecker"]["GraphCodeBERT"].keys():
                            train_other_trafo_test_trafo_GraphCodeBERT.append(reduce(get_score_list(results["VulDeePecker"]["GraphCodeBERT"][other_trafo][transformation], metric=metric)))


        fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH, FIG_HEIGHT))
        
        clean_accuracy = reduce(get_score_list(results["VulDeePecker"]["UniXcoder"]["no_transformation"]["no_transformation"], metric=metric))
        ax.plot(0.85, clean_accuracy, marker="*", color="gray", markersize=6.2)
        ax.axhline(clean_accuracy, color='gray', linestyle=":", linewidth=0.8)

        clean_accuracy = reduce(get_score_list(results["VulDeePecker"]["CoTexT"]["no_transformation"]["no_transformation"], metric=metric))
        ax.plot(1.85, clean_accuracy, marker="*", color="gray", markersize=6.2)
        ax.axhline(clean_accuracy, color='gray', linestyle=":", linewidth=0.8)
        
        clean_accuracy = reduce(get_score_list(results["VulDeePecker"]["GraphCodeBERT"]["no_transformation"]["no_transformation"], metric=metric))
        ax.plot(2.85, clean_accuracy, marker="*", color="gray", markersize=6.2)
        ax.axhline(clean_accuracy, color='gray', linestyle=":", linewidth=0.8)
        
        clean_accuracy = reduce(get_score_list(results["VulDeePecker"]["CodeBERT"]["no_transformation"]["no_transformation"], metric=metric))
        ax.plot(3.85, clean_accuracy, marker="*", color="gray", markersize=6.2)
        ax.axhline(clean_accuracy, color='gray', linestyle=":", linewidth=0.8)

        clean_accuracy = reduce(get_score_list(results["VulDeePecker"]["VulBERTa"]["no_transformation"]["no_transformation"], metric=metric))
        ax.plot(4.85, clean_accuracy, marker="*", color="gray", markersize=6.2)
        ax.axhline(clean_accuracy, color='gray', linestyle=":", linewidth=0.8)
        
        clean_accuracy = reduce(get_score_list(results["VulDeePecker"]["PLBart"]["no_transformation"]["no_transformation"], metric=metric))
        ax.plot(5.85, clean_accuracy, marker="*", color="gray", markersize=6.2)
        ax.axhline(clean_accuracy, color='gray', linestyle=":", linewidth=0.8)
        

        ax.set_yticks(METRIC_TO_XTICKS[metric])

        #ax.set_xlim([0, 11])
        ax.set_ylim(METRIC_TO_YLIM[metric])

        ax.set_ylabel("test set {}".format(metric))

        handles = []
        labels = []

        bp = ax.boxplot(x=[train_no_trafo_test_trafo_UniXcoder, train_trafo_test_trafo_UniXcoder, train_other_trafo_test_trafo_UniXcoder, train_no_trafo_test_trafo_cotext, train_trafo_test_trafo_cotext, train_other_trafo_test_trafo_cotext, train_no_trafo_test_trafo_GraphCodeBERT, train_trafo_test_trafo_GraphCodeBERT, train_other_trafo_test_trafo_GraphCodeBERT, train_no_trafo_test_trafo_CodeBERT, train_trafo_test_trafo_CodeBERT, train_other_trafo_test_trafo_CodeBERT, train_no_trafo_test_trafo_vulberta, train_trafo_test_trafo_vulberta, train_other_trafo_test_trafo_vulberta, train_no_trafo_test_trafo_plbart, train_trafo_test_trafo_plbart, train_other_trafo_test_trafo_plbart],  # sequence of arrays
        positions=[0.7, 1.0, 1.2, 1.7, 2.0, 2.2, 2.7, 3.0, 3.2, 3.7, 4.0, 4.2, 4.7, 5.0, 5.2, 5.7, 6.0, 6.2],   # where to put these arrays
        widths=(0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13),
        patch_artist=True, sym="")
        
        for median in bp['medians']:
            median.set_color(MEDIAN_LINE_COLOR)

        ax.set_xticks([1, 2, 3, 4, 5, 6])
        ax.set_xticklabels(["UniXcoder", "CoTexT", "GraphCB", "CodeBERT", "VulBERTa", "PLBart"],rotation=0, fontsize=X_TICK_LABELS_FONT_SIZE)

        patch_colors = [COLORS[0], COLORS[1], COLORS[2], COLORS[0], COLORS[1], COLORS[2], COLORS[0], COLORS[1], COLORS[2], COLORS[0], COLORS[1], COLORS[2], COLORS[0], COLORS[1], COLORS[2], COLORS[0], COLORS[1], COLORS[2]]
        for patch, color in zip(bp['boxes'], patch_colors):
                patch.set_facecolor(color)

                
        from matplotlib.lines import Line2D
        handles.append(Line2D([0], [0], color='gray', linestyle=":", marker="*", markersize=6.2, linewidth=0.8))
        labels.append("Train: $Tr$, Test: $Te$")
        handles.append(Line2D([0], [0], color=COLORS[0], lw=0.0, marker="s", markersize=8))
        labels.append("Train: $Tr$, Test: $Te_k$")
        handles.append(Line2D([0], [0], color=COLORS[1], lw=0.0, marker="s", markersize=8))
        labels.append("Train: $Tr_k$, Test: $Te_k$")
        handles.append(Line2D([0], [0], color=COLORS[2], lw=0.0, marker="s", markersize=8))
        labels.append("Train: $Tr_k$, Test: $Te_{j \\neq k}$")

        if (METRIC_TO_LEGEND[metric]):
            legend = ax.legend(handles=handles, labels=labels,loc='upper right', frameon=True)
            legend.get_frame().set_alpha(1)

        fig.tight_layout(pad=0.02)

        figname = inspect.stack()[0][3] + "_" + metric
        plt.savefig("{}{}.pdf".format(OUT_DIR, figname), dpi=DPI, format="pdf")

    
def fig_7_a():
    
    results = load_results()

    x = np.arange(1,11,1)

    fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH, FIG_HEIGHT))

    ax.set_xticks(x)
    ax.set_yticks(np.arange(0,1,0.02))

    #ax.set_xlim([0, 11])
    ax.set_ylim([0.465, 0.72])

    ax.set_xlabel("training epoch")
    ax.set_ylabel("test set accuracy")

    #ax.yaxis.grid(color='gray')

    
    handles = []
    labels = []
    
    for y in np.arange(0.48, 0.7, 0.02):
        ax.axhline(y = y, color = 'lightgray', linestyle = '-', lw=0.5)
    
    for i, trafo in enumerate(SEMANTIC_PRESERVING_TRANSFORMATIONS):
        if trafo != "no_transformation" and trafo != "tf_11" and trafo != "tf_10":
            p = ax.plot(x, get_score_list(results["CodeXGLUE"]["VulBERTa"][trafo]["tf_10"]), color=COLORS[2], linewidth=0.5)
            

    p = ax.plot(x, get_score_list(results["CodeXGLUE"]["VulBERTa"]["no_transformation"]["no_transformation"]), color = "gray", marker="o", markersize=4.8, linestyle=":")
    p = ax.plot(x, get_score_list(results["CodeXGLUE"]["VulBERTa"]["no_transformation"]["tf_10"]), marker='o', color = COLORS[0], markersize=4.8, linestyle="-.")
    p = ax.plot(x, get_score_list(results["CodeXGLUE"]["VulBERTa"]["tf_10"]["tf_10"]), marker='o', color = COLORS[1], markersize=4.8)
#    p = ax.plot(x, get_score_list(results["CodeXGLUE"]["VulBERTa"]["RG"]["no_transformation"]), marker='o', color = "#000000", linestyle="--", markersize=4.8)
            
    
    from matplotlib.lines import Line2D
    handles.append(Line2D([0], [0], color='gray', linestyle=":"))
    labels.append("Train: $Tr$, Test: $Te$")
    handles.append(Line2D([0], [0], color = COLORS[0], linestyle="-."))
    labels.append("Train: $Tr$, Test: $Te_{10}$")
    handles.append(Line2D([0], [0], color = COLORS[1]))
    labels.append("Train: $Tr_{10}$, Test: $Te_{10}$")
    handles.append(Line2D([0], [0], color = "#000000", linestyle="--"))
    labels.append("Random Guessing")
    
                            
    handles.append(Line2D([0], [0], color=COLORS[2], lw=0.7))
    labels.append("Train: $Tr_{j \\neq 10}$, Test: $Te_{10}$")
            
    legend = ax.legend(ncol=2, handles=handles, labels=labels,loc='upper left', frameon=True)

    fig.tight_layout(pad=0.02)

    figname = inspect.stack()[0][3]
    plt.savefig("{}{}.pdf".format(OUT_DIR, figname), dpi=DPI, format="pdf")
    
def table_2():
    
    results = load_results()

    datasets = ["CodeXGLUE", "VulDeePecker"]
    
    s_Tr_Te = dict()
    s_Tr_Te_k = dict()
    s_Tr_k_Te_k = dict()
    s_Tr_j_Te_k = dict()
    
    for metric in METRICS:
    
        s_Tr_Te[metric] = dict()
        s_Tr_Te_k[metric] = dict()
        s_Tr_k_Te_k[metric] = dict()
        s_Tr_j_Te_k[metric] = dict()
        
        for dataset in datasets:
            
            s_Tr_Te[metric][dataset] = dict()
            s_Tr_Te_k[metric][dataset] = dict()
            s_Tr_k_Te_k[metric][dataset] = dict()
            s_Tr_j_Te_k[metric][dataset] = dict()
            
            for technique in TECHNIQUES:
                
        
                s_Tr_Te[metric][dataset][technique] = reduce(get_score_list(results[dataset][technique]["no_transformation"]["no_transformation"], metric=metric), metric=metric)
                
                s_Tr_Te_k[metric][dataset][technique] = []
                
                s_Tr_k_Te_k[metric][dataset][technique] = []
                
                s_Tr_j_Te_k[metric][dataset][technique] = []
                
        
    for transformation in SEMANTIC_PRESERVING_TRANSFORMATIONS:
        if transformation != "no_transformation":
            
            for metric in METRICS:
                for dataset in datasets:
                    for technique in TECHNIQUES:
                        if transformation in results[dataset][technique]["no_transformation"].keys():
                            s_Tr_Te_k[metric][dataset][technique].append(reduce(get_score_list(results[dataset][technique]["no_transformation"][transformation], metric=metric), metric=metric) - s_Tr_Te[metric][dataset][technique])
                        if transformation in results[dataset][technique].keys():
                            s_Tr_k_Te_k[metric][dataset][technique].append(reduce(get_score_list(results[dataset][technique][transformation][transformation], metric=metric), metric=metric) - s_Tr_Te[metric][dataset][technique])

                for other_trafo in SEMANTIC_PRESERVING_TRANSFORMATIONS:
                    if other_trafo != transformation and other_trafo != "tf_11":
                        for metric in METRICS:
                            for dataset in datasets:
                                for technique in TECHNIQUES:
                                    if other_trafo in results[dataset][technique].keys() and transformation in results[dataset][technique][other_trafo].keys():
                                        s_Tr_j_Te_k[metric][dataset][technique].append(reduce(get_score_list(results[dataset][technique][other_trafo][transformation], metric=metric), metric=metric) - s_Tr_Te[metric][dataset][technique])

                        
                
        
    figname = inspect.stack()[0][3]
    with open("{}{}.txt".format(OUT_DIR, figname), "w") as file:
        lines = []
        average_restorations = dict()
        average_further_drops = dict()
        
        for metric in METRICS:
            
            scores = []
            average_restorations[metric] = dict()
            average_further_drops[metric] = dict()
            
            for technique in TECHNIQUES:        
                
                average_restorations[metric]["CodeXGLUE"] = []
                average_further_drops[metric]["CodeXGLUE"] = []
                average_restorations[metric]["VulDeePecker"] = []
                average_further_drops[metric]["VulDeePecker"] = []
            
                metric_label = metric if technique == "UniXcoder" else ""
                
                scores.append([np.array(s_Tr_Te[metric]["CodeXGLUE"][technique]).mean(), 
                            np.array(s_Tr_Te_k[metric]["CodeXGLUE"][technique]).mean(), 
                            np.array(s_Tr_k_Te_k[metric]["CodeXGLUE"][technique]).mean(), 
                            np.array(s_Tr_j_Te_k[metric]["CodeXGLUE"][technique]).mean(),
                            np.array(s_Tr_Te[metric]["VulDeePecker"][technique]).mean(), 
                            np.array(s_Tr_Te_k[metric]["VulDeePecker"][technique]).mean(), 
                            np.array(s_Tr_k_Te_k[metric]["VulDeePecker"][technique]).mean(), 
                            np.array(s_Tr_j_Te_k[metric]["VulDeePecker"][technique]).mean()])
                
                if metric not in ["FPR", "FNR"]:
                    lines.append("{} & {} & {:.3f} & {:.3f} $\\downarrow$ & {:.3f} $\\uparrow$ & {:.3f} $\\downarrow$ & {:.3f} & {:.3f} $\\downarrow$ & {:.3f} $\\uparrow$ & {:.3f} $\\downarrow$ \\\\ \n".format(metric_label, 
                                                                                                    technique, 
                                                                                                    round(np.array(s_Tr_Te[metric]["CodeXGLUE"][technique]).mean(), 3), 
                                                                                                    round(np.array(s_Tr_Te_k[metric]["CodeXGLUE"][technique]).mean(), 3), 
                                                                                                    round(np.array(s_Tr_k_Te_k[metric]["CodeXGLUE"][technique]).mean(), 3), 
                                                                                                    round(np.array(s_Tr_j_Te_k[metric]["CodeXGLUE"][technique]).mean(), 3),
                                                                                                    round(np.array(s_Tr_Te[metric]["VulDeePecker"][technique]).mean(), 3), 
                                                                                                    round(np.array(s_Tr_Te_k[metric]["VulDeePecker"][technique]).mean(), 3), 
                                                                                                    round(np.array(s_Tr_k_Te_k[metric]["VulDeePecker"][technique]).mean(), 3), 
                                                                                                    round(np.array(s_Tr_j_Te_k[metric]["VulDeePecker"][technique]).mean(), 3)
                                                                                                    ))
                else:
                    lines.append("{} & {} & {:.3f} & {:.3f} $\\uparrow$ & {:.3f} $\\downarrow$ & {:.3f} $\\uparrow$ & {:.3f} & {:.3f} $\\uparrow$ & {:.3f} $\\downarrow$ & {:.3f} $\\uparrow$ \\\\ \n".format(metric_label, 
                                                                                                    technique, 
                                                                                                    round(np.array(s_Tr_Te[metric]["CodeXGLUE"][technique]).mean(), 3), 
                                                                                                    round(np.array(s_Tr_Te_k[metric]["CodeXGLUE"][technique]).mean(), 3), 
                                                                                                    round(np.array(s_Tr_k_Te_k[metric]["CodeXGLUE"][technique]).mean(), 3), 
                                                                                                    round(np.array(s_Tr_j_Te_k[metric]["CodeXGLUE"][technique]).mean(), 3),
                                                                                                    round(np.array(s_Tr_Te[metric]["VulDeePecker"][technique]).mean(), 3), 
                                                                                                    round(np.array(s_Tr_Te_k[metric]["VulDeePecker"][technique]).mean(), 3), 
                                                                                                    round(np.array(s_Tr_k_Te_k[metric]["VulDeePecker"][technique]).mean(), 3), 
                                                                                                    round(np.array(s_Tr_j_Te_k[metric]["VulDeePecker"][technique]).mean(), 3)
                                                                                                    ))
                    
                average_restorations[metric]["CodeXGLUE"].append(1 - (round(np.array(s_Tr_k_Te_k[metric][dataset][technique]).mean(), 3) / round(np.array(s_Tr_Te_k[metric]["CodeXGLUE"][technique]).mean(), 3)))
                average_further_drops[metric]["CodeXGLUE"].append((round(np.array(s_Tr_j_Te_k[metric][dataset][technique]).mean(), 3) / round(np.array(s_Tr_Te_k[metric]["CodeXGLUE"][technique]).mean(), 3)) - 1)
                average_restorations[metric]["VulDeePecker"].append(1 - (round(np.array(s_Tr_k_Te_k[metric][dataset][technique]).mean(), 3) / round(np.array(s_Tr_Te_k[metric]["VulDeePecker"][technique]).mean(), 3)))
                average_further_drops[metric]["VulDeePecker"].append((round(np.array(s_Tr_j_Te_k[metric][dataset][technique]).mean(), 3) / round(np.array(s_Tr_Te_k[metric]["VulDeePecker"][technique]).mean(), 3)) - 1)

            lines[-1] = lines[-1][:-2]
            lines[-1] = lines[-1] + " \\hline\n"
            scores = np.array(scores)
            
            if metric not in ["FPR", "FNR"]:
                lines.append(" &  &  & \\textbf{{{:.3f}}} $\\downarrow$ & \\textbf{{{:.3f}}} $\\uparrow$ & \\textbf{{{:.3f}}} $\\downarrow$ &  & \\textbf{{{:.3f}}} $\\downarrow$ & \\textbf{{{:.3f}}} $\\uparrow$ & \\textbf{{{:.3f}}} $\\downarrow$ \\\\ \n".format(
                                                                                                round(scores[:, 1].mean(), 3),
                                                                                                round(scores[:, 2].mean(), 3),
                                                                                                round(scores[:, 3].mean(), 3),
                                                                                                round(scores[:, 5].mean(), 3),
                                                                                                round(scores[:, 6].mean(), 3),
                                                                                                round(scores[:, 7].mean(), 3)
                                                                                                ))
            else:
                lines.append(" &  &  & \\textbf{{{:.3f}}} $\\uparrow$ & \\textbf{{{:.3f}}} $\\downarrow$ & \\textbf{{{:.3f}}} $\\uparrow$ &  & \\textbf{{{:.3f}}} $\\uparrow$ & \\textbf{{{:.3f}}} $\\uparrow$ & \\textbf{{{:.3f}}} $\\uparrow$ \\\\ \n".format(
                                                                                                round(scores[:, 1].mean(), 3),
                                                                                                round(scores[:, 2].mean(), 3),
                                                                                                round(scores[:, 3].mean(), 3),
                                                                                                round(scores[:, 5].mean(), 3),
                                                                                                round(scores[:, 6].mean(), 3),
                                                                                                round(scores[:, 7].mean(), 3)
                                                                                                ))
                
            if metric != "FNR":
                lines[-1] = lines[-1][:-2]
                lines[-1] = lines[-1] + " \\hline\n"
            
        
        file.writelines(lines)
        
def table_3():
    
    results = load_results()

    dataset = "CodeXGLUE"
    
    s_Tr_Te = dict()
    s_Tr_VPTe = dict()
    s_VPTr_VPTe = dict()
    s_VPTr_Te= dict()
    
    for metric in ["accuracy", "f1", "recall", "precision", "FPR", "FNR"]:
        
        s_Tr_Te[metric] = dict()
        s_Tr_VPTe[metric] = dict()
        s_VPTr_VPTe[metric] = dict()
        s_VPTr_Te[metric]= dict()
    
        for technique in TECHNIQUES:
        
            s_Tr_Te[metric][technique] = reduce(get_score_list(results[dataset][technique]["no_transformation"]["no_transformation"], metric=metric), metric=metric)
            s_Tr_VPTe[metric][technique] = reduce(get_score_list(results[dataset][technique]["no_transformation"]["fixed_nonfixed"], metric=metric), metric=metric)
            s_VPTr_VPTe[metric][technique] = reduce(get_score_list(results["VulnPatchPairs"][technique], metric=metric, prefix = "vpp/"), metric=metric)
            s_VPTr_Te[metric][technique] = reduce(get_score_list(results["VulnPatchPairs"][technique], metric=metric, prefix = "codexglue/"), metric=metric)
            
    
    figname = inspect.stack()[0][3]
    with open("{}{}.txt".format(OUT_DIR, figname), "w") as file:
        lines = []
        
        for metric in ["accuracy", "f1", "recall", "precision", "FPR", "FNR"]:

            scores = []
        
            for technique in TECHNIQUES:
                
                metric_label = metric if technique == "UniXcoder" else ""
                
                scores.append([s_Tr_Te[metric][technique], 
                                s_Tr_VPTe[metric][technique],
                                s_VPTr_VPTe[metric][technique],
                                s_VPTr_Te[metric][technique]
                            ])
                
                lines.append("{} & {} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\ \n".format(          metric_label,
                                                                                                    technique, 
                                                                                                    s_Tr_Te[metric][technique], 
                                                                                                    s_Tr_VPTe[metric][technique],
                                                                                                    s_VPTr_VPTe[metric][technique],
                                                                                                    s_VPTr_Te[metric][technique],
                                                                                                    ))
            lines[-1] = lines[-1][:-2]
            lines[-1] = lines[-1] + " \\hline\n"

            scores = np.array(scores)   
            lines.append(" & & \\textbf{{{:.3f}}} & \\textbf{{{:.3f}}} & \\textbf{{{:.3f}}} & \\textbf{{{:.3f}}}\\\\ \n".format(round(scores[:, 0].mean(), 3),
                                                                        round(scores[:, 1].mean(), 3),
                                                                        round(scores[:, 2].mean(), 3),
                                                                        round(scores[:, 3].mean(), 3)
                                                                        ))

            if metric != "FNR":
                lines[-1] = lines[-1][:-2]
                lines[-1] = lines[-1] + " \\hline\n"
        
        file.writelines(lines)
        
        
def main(params):
    
    fig_functions = [
        fig_5_a,
        fig_5_b,
        fig_6,
        fig_7_a,
        fig_7_b,
        fig_7_c,
        fig_7_d,
        table_2,
        table_3,
        fig_8,
        fig_9,
        fig_10,
        fig_11
    ]
    
    progress_bar = tqdm(range(len(fig_functions)))
    
    for func in fig_functions:
        func()
        progress_bar.update(1)

if __name__ == '__main__':
    params=parse_args()
    main(params)
# Uncovering the Limits of Machine Learning for Automatic Vulnerability Detection (USENIX 2024)

## Structure

Below is an annotated map of the directory structure of this repository.

```
.
├── scripts................................ Scripts to exactly reproduce all experiments presented in our paper.
│   └── <dataset>.......................... One directory for each dataset (CodeXGLUE, VulDeePecker).
│       └── <technique>.................... One directory for each ML technique (VulBERTa, CoTexT, PLBart)
│           ├──  run.py.................... Compute experimental results for selected ML technique and dataset.
│           └──  run.sbatch................ SBATCH script for running run.py on a GPU cluster.
│
├── datasets............................... All datasets that we use in the experiments for the paper + our own created dataset VulnPatchPairs.
│   └── README.md.......................... Instructions for downloading all datasets used in this repository.
│
├── models................................. All pretrained models that are not downloaded in the training scripts.
│   └── README.md.......................... Instructions for downloading all models used in this repository.
│
├── plots.................................. Generates all plots shown in the paper.
│   ├── generate_plots.py.................. Script that generates all plots from the experimental results.
│   └── plots.............................. The produced plots and tables that can be found in the paper.
│
├── install_requirements.sh................ Script to install Python environment and required packages.
├── requirements.txt....................... All Python packages that you need to run the experiments.
│
└── README.md
```

## Setup

### Step 1: Install Anaconda

Anaconda is an open-source package and environment management tool for Python. Instructions for Installation can be found [here](https://www.anaconda.com/products/distribution).

### Step 2: Install Requirements

We assume that you have Anaconda installed.

Running the following script from the root directory of this repository creates a virtual environment in Anaconda, and installs the required Python packages.

```
bash install_requirements.sh
```

Activate the environment with the following command.

```
conda activate LimitsOfMl4Vuln
```

### Step 3: Download the required datasets

Go to [datasets/README.md](https://github.com/niklasrisse/USENIX_2024/blob/main/datasets/README.md) and follow the instructions to download all datasets needed to run our experiments.

### Step 4: Download the required models

Go to [models/README.md](https://github.com/niklasrisse/USENIX_2024/blob/main/models/README.md) and follow the instructions to download all models needed to run our experiments.

### Step 5: Ready to go

## How to reproduce our figures

In order to reproduce our figures, you need to run [plots/generate_plots.py](https://github.com/niklasrisse/USENIX_2024/blob/main/plots/generate_plots.py). Navigate to the root directly of this repository, and run:

```
python plots/generate_plots.py
```

After running the script, you will find the reproduced figures in the [plots/plots](https://github.com/niklasrisse/USENIX_2024/blob/main/plots/plots) directory.

## How to reproduce our results

In order to reproduce our results, you first need to complete the setup described above. After that, you need to run all python scripts in the [scripts](https://github.com/niklasrisse/USENIX_2024/blob/main/scripts) directory and its subdirectories. While it is theoretically possible to run this on a local machine (e.g. MacBook), we do not recommend this. We ran all our scripts on a GPU cluster using 4 Nvidia A100 40GB-SXM GPUs, each one equipped with 512 GB of RAM. If you have access to a GPU cluster which runs on SLURM, you can use the .sbatch scripts that we provided in each of the subdirectories. After computing all of the results, you can re-run [plots/generate_plots.py](https://github.com/niklasrisse/USENIX_2024/blob/main/plots/generate_plots.py) to see if your reproduced results match our results.

## Citation

If you want to use our work, please use the following citation. We will update this to the USENIX citation, once it is published.

```
@misc{risse2023limits,
      title={Limits of Machine Learning for Automatic Vulnerability Detection},
      author={Niklas Risse and Marcel Böhme},
      year={2023},
      eprint={2306.17193},
      archivePrefix={arXiv},
      primaryClass={cs.CR}
}
```

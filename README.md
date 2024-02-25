# Annotability
Annotatability, a method to identify meaningful patterns in single-cell genomics data through annotation-trainability analysis, which estimates annotation congruence using a rich but often overlooked signal, namely the training dynamics of a deep neural network. 

![workflow](https://github.com/nitzanlab/Annotatability/blob/main/fig1.png?raw=true)

<!-- Manuscript -->
## Manuscript

<!-- GETTING STARTED -->
## Getting Started

<!-- Reproducibility -->
## Reproducibility
<h4> For reproducibility of Annotability manuscript, please refer to:<br /> https://github.com/nitzanlab/Annotatability_notebooks</h4>

<!-- Installation -->
## Installation

```sh
git clone https://github.com/nitzanlab/Annotatability.git
cd Annotatability
pip install .
```
<br />

<!-- Tests -->

## How to use
We strongly recommend utilizing ScanPy [Scanpy](https://scanpy.readthedocs.io/en/stable/) for the analysis of scRNA-seq data. <br />
Annotatability comprises two code files:<br /> "models.py," which encompasses the training of neural network functions and the generation of the trainability-aware graph.<br />
"metrics.py," which contains the scoring functions.<br />
Imports:
```
from Annotatability import metrics, models
import numpy as np
import scanpy as sc
from torch.utils.data import TensorDataset, DataLoader , WeightedRandomSampler
import torch
import torch.optim as optim
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
```

Given annotated data (of type Anndata) named "adata", and the annotation "label" we aim to analyze(stores as observation). <br />
For estimate the confidence and the variability of the annotation for each cell (and store the as observation) we will use the following commands:
```
epoch_num=50 %Can be changed
prob_list = models.follow_training_dyn_neural_net(adata, label_key='label',iterNum=epoch_num)
all_conf , all_var = models.probability_list_to_confidence_and_var(prob_list, n_obs= adata.n_obs, epoch_num=epoch_num, device=device)
adata.obs["var"] = all_var.detach().numpy()
adata.obs["conf"] = all_conf.detach().numpy()
```
For computing the 

## Running the tests


```
conda create -n Annotatability python=3.10
conda activate Annotatability
pip install .
pytest tests/test
```

<!-- CONTACT -->
## Contact
Jonathan Karin - jonathan.karin [at ] mail.huji.ac.il <br />

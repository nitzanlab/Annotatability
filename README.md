# Annotatability
Annotatability is a method that identifies meaningful patterns in single-cell genomics data through annotation-trainability analysis; It estimates congruence in annotation structure by analyzing the training dynamics of deep neural networks.<br />
Annotatability can be used to audit and rectify erroneous cell annotations, identify intermediate cell states, delineate complex temporal trajectories along development, characterize cell diversity in diseased tissue, identify disease-related genes, assess treatment effectiveness, and identify rare healthy-like cell populations.

![workflow](https://github.com/nitzanlab/Annotatability/blob/main/fig1.png?raw=true)

<!-- Manuscript -->
## Manuscript
Karin, J., Mintz, R., Raveh, B. et al. Interpreting single-cell and spatial omics data using deep neural network training dynamics. Nat Comput Sci (2024). https://doi.org/10.1038/s43588-024-00721-5



<!-- Reproducibility -->
## Reproducibility
<h4> For reproducibility of the results presented in the Annotability manuscript, please refer to:<br /> https://github.com/nitzanlab/Annotatability_notebooks</h4>

<!-- Installation -->
## Installation
```sh
pip install Annotatability
```
<br />
<!-- Tests -->

## How to use
Examples of the usage of our method are available in the reproducibility repository ( https://github.com/nitzanlab/Annotatability_notebooks ) and for two additional manuscript which are not included in the manuscript: [tutorial1](https://github.com/nitzanlab/Annotatability_notebooks/blob/main/tutorial_retina.ipynb) (runtime of a few minutes with GPU, ~20 minutes without GPU)- finding erroneous annotations and intermediate cell states in retina bipolar cells, or [tutorial2](https://github.com/nitzanlab/Annotatability_notebooks/blob/main/tutorial_covid.ipynb)  - analysis of case-control dataset of COVID-19. <br />
Our code is based on [Scanpy](https://scanpy.readthedocs.io/en/stable/) package, but it can be easily adapted for any Python-based single-cell data analysis. Annotatability comprises two code files:<br /> models.py, which encompasses the training of neural network functions and the generation of the trainability-aware graph.<br />
metrics.py, which contains the scoring functions.<br />
<b>Imports</b>:<br />
```
from Annotatability import metrics, models
import numpy as np
import scanpy as sc
from torch.utils.data import TensorDataset, DataLoader , WeightedRandomSampler
import torch
import torch.optim as optim
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
```

<b>Train the neural network and calculate the confidence and variability metrics</b><br />
We take as input annotated data (of type Anndata) named “adata”, and the annotation “label” (stores as observation) we aim to analyze.<br />
To estimate the confidence and variability of the annotation of each cell, we  use the following commands:
```
epoch_num=50 %Can be changed
prob_list = models.follow_training_dyn_neural_net(adata, label_key='label',iterNum=epoch_num, device=device)
all_conf , all_var = models.probability_list_to_confidence_and_var(prob_list, n_obs= adata.n_obs, epoch_num=epoch_num)
adata.obs["var"] = all_var.detach().numpy()
adata.obs["conf"] = all_conf.detach().numpy()
```
For 'follow_training_dyn_neural_net' function, we can change the following hyperparameters- <br />
```
    iterNum : int, optional (default=100)
        Number of training iterations (epochs).

    lr : float, optional (default=0.001)
        Learning rate for the optimizer.

    momentum : float, optional (default=0.9)
        Momentum for the optimizer.

    device : str, optional (default='cpu')
        Device for training the neural network ('cpu' or 'cuda' for GPU).

    weighted_sampler : bool, optional (default=True)
        Whether to use a weighted sampler for class imbalance.

    batch_size : int, optional (default=256)
        Batch size for training.
    num_layers : int, optional (default=3)
        Depth of the neural network. Values alowed=3/4/5
```
<b> Compute the annotation-trainability score</b>
```
adata_ranked = metrics.rank_genes_conf_min_counts(adata)
```
The results will be stored as variables in:
```
adata_ranked.var['conf_score_high'] %annotation-trainability positive association score
adata_ranked.var['conf_score_low'] %annotation-trainability negative association score
```

<b> Trainability-aware graph embedding</b> 
```
import scipy.sparse as sp
connectivities_graph , distance_graph  = metrics.make_conf_graph(adata.copy(), alpha=0.9 , k=15)
adata.obsp['connectivities']=sp.csr_matrix(connectivities_graph)
```
Note: 'alpha' can be adjusted.<br />
To visualize of the trainability-aware graph, use the following functions:

```
sc.tl.umap(adata)
sc.pl.umap(adata, color='conf')
```
Notice that using sc.pp.neighbors(adata) will store the neighbors graph in adata.obsp['connectivities'] instead of the trainability-aware graph.

## System requirements
```
python (>3.0)
```
packages:
```
"numpy",
"scanpy",
"numba",
"pandas",
"scipy",
"matplotlib",
"pytest",
"torch"
```

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

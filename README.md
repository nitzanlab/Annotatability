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
Given an annotated data (of type Anndata) named "adata".
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

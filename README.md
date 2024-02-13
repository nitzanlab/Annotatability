# Annotability
Annotatability, a method to identify meaningful patterns in single-cell genomics data through annotation-trainability analysis, which estimates annotation congruence using a rich but often overlooked signal, namely the training dynamics of a deep neural network. 

<!-- Manuscript -->
## Manuscript

<!-- GETTING STARTED -->
## Getting Started

<!-- Reproducibility -->
## Reproducibility
<h4> For reproducibility of Annotability manuscript, please refer to:<br /> [https://github.com/nitzanlab/Annotatability_notebooks](https://github.com/nitzanlab/Annotatability_notebooks)</h4>

<!-- Installation -->
## Installation

```sh
git clone https://github.com/nitzanlab/Annotatability.git
cd Annotatability
pip install .
```
<br />

<!-- Tests -->

## Running the tests



### Running the gpu only tests 

```
conda create -n Annotatability python=3.10
conda activate Annotatability
pip install .
pytest tests/test
```

<!-- CONTACT -->
## Contact
Jonathan Karin - jonathan.karin [at ] mail.huji.ac.il <br />

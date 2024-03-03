# AdapterGNN: Parameter-Efficient Fine-Tuning Improves Generalization in GNNs


## Installation
We used the following Python packages for core development. We tested on `Python 3.9`.
```
torch                 1.10.2
torch-cluster         1.5.9
torch-geometric       2.0.3
torch-scatter         2.0.9
torch-sparse          0.6.12
torch-spline-conv     1.2.1
rdkit                 2020.09.1
```

You can use conda to install pytorch first and then use `environment.yml` to automatically install the required packages:
```
conda env create -f environment.yml
```

## Dataset download
All the necessary data files can be downloaded from the following links.

For the chemistry dataset, download from [chem data](https://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip) (2.5GB), unzip it, and put it under `chem/`.

For the biology dataset, download from [bio data](https://snap.stanford.edu/gnn-pretrain/data/bio_dataset.zip) (2GB), unzip it, and put it under `bio/`.

## Pre-training

Due to size limits, we cannot include the pre-trained checkpoints in this repo. Please refer to:

https://github.com/snap-stanford/pretrain-gnns

https://github.com/Shen-Lab/GraphCL

https://github.com/mpanpan/SimGRACE

for these pre-trained checkpoints: 

```contextpred.pth infomax.pth edgepred.pth masking.pth graphcl_80.pth simgrace_80.pth```

And put these pre-trained checkpoints in bio/model_gin and chem/model_gin.

## Reproducing results in the paper
Our results in the paper can be reproduced by:

- `cd chem`
  
- `sh reproduce.sh`

and 

- `cd bio`
  
- `sh reproduce.sh`

The results will be automatically recorded into `log/` directory 

To compare with fine-tuning baseline, you can manually replace `AdapterGNN_graphpred` with `GNN_graphpred` in `finetune.py`

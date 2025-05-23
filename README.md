# CRSLab

[![Pypi Latest Version](https://img.shields.io/pypi/v/crslab)](https://pypi.org/project/crslab)
[![Release](https://img.shields.io/github/v/release/rucaibox/crslab.svg)](https://github.com/rucaibox/crslab/releases)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-CRSLab-%23B21B1B)](https://arxiv.org/abs/2101.00939)
[![Documentation Status](https://readthedocs.org/projects/crslab/badge/?version=latest)](https://crslab.readthedocs.io/en/latest/?badge=latest)

[Paper](https://arxiv.org/pdf/2101.00939.pdf) | [Docs](https://crslab.readthedocs.io/en/latest/?badge=latest)
| [中文版](./README_CN.md)

**CRSLab** is an open-source toolkit for building Conversational Recommender System (CRS). It is developed based on
Python and PyTorch. CRSLab has the following highlights:

- **Comprehensive benchmark models and datasets**: We have integrated commonly-used 6 datasets and 18 models, including graph neural network and pre-training models such as R-GCN, BERT and GPT-2. We have preprocessed these datasets to support these models, and release for downloading.
- **Extensive and standard evaluation protocols**: We support a series of widely-adopted evaluation protocols for testing and comparing different CRS.
- **General and extensible structure**: We design a general and extensible structure to unify various conversational recommendation datasets and models, in which we integrate various built-in interfaces and functions for quickly development.
- **Easy to get started**: We provide simple yet flexible configuration for new researchers to quickly start in our library. 
- **Human-machine interaction interfaces**: We provide flexible human-machine interaction interfaces for researchers to conduct qualitative analysis.

<p align="center">
  <img src="https://i.loli.net/2020/12/30/6TPVG4pBg2rcDf9.png" alt="RecBole v0.1 architecture" width="400">
  <br>
  <b>Figure 1</b>: The overall framework of CRSLab
</p>




- [CRSLab](#crslab)
  - [Installation](#installation)
    - [Install PyTorch](#install-pytorch)
    - [Install PyTorch Geometric](#install-pytorch-geometric)
    - [Install CRSLab](#install-crslab)
  - [Quick-Start](#quick-start)
  - [Models](#models)
  - [Datasets](#datasets)
  - [Performance](#performance)
    - [Recommendation Task](#recommendation-task)
    - [Conversation Task](#conversation-task)
    - [Policy Task](#policy-task)
  - [Team](#team)
  - [License](#license)



## Installation

Currently, this version can only run on Linux and has updated most dependencies to their latest versions (as of 2025.5.19).

### Install PyTorch

Use PyTorch [local installation](https://pytorch.org/get-started/locally/) or [previous versions installation](https://pytorch.org/get-started/previous-versions/) commands to install PyTorch. For example, on Linux:

```bash
conda create -n crslab python=3.13
conda activate crslab
conda install gcc_linux-64 gxx_linux-64 cmake pip # cmake and pip are required for 3.13
conda install -c conda-forge gperftools # required for 3.13
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
```

If you want to use CRSLab with GPU, make sure the following command prints `True` after installation:

```bash
$ python -c "import torch; print(torch.cuda.is_available())"
>>> True
```

### Install PyTorch Geometric

Ensure that at least PyTorch 1.8.0 is installed:

```bash
$ python -c "import torch; print(torch.__version__)"
>>> 1.8.0
```

Find the CUDA version PyTorch was installed with:

```bash
$ python -c "import torch; print(torch.version.cuda)"
>>> 12.6
```

For Linux:

Install the relevant packages:

```bash
pip install torch_geometric
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.0+cu126.html
```

For other systems:

Check PyG [installation documents](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) to install the relevant packages.

### Install CRSLab

Install from source:

```bash
git clone https://github.com/GlorYouth/CRSLab.git && cd CRSLab
pip install -r requirements.txt
pip install -e .
```



## Quick-Start

With the source code, you can use the provided script for initial usage of our library with cpu by default:

```bash
python run_crslab.py --config config/crs/kgsf/redial.yaml
```

The system will complete the data preprocessing, and training, validation, testing of each model in turn. Finally it will get the evaluation results of specified models.

If you want to save pre-processed datasets and training results of models, you can use the following command:

```bash
python run_crslab.py --config config/crs/kgsf/redial.yaml --save_data --save_system
```

In summary, there are following arguments in `run_crslab.py`:

- `--config` or `-c`: relative path for configuration file(yaml).
- `--gpu` or `-g`: specify GPU id(s) to use, we now support multiple GPUs. Defaults to CPU(-1).
- `--save_data` or `-sd`: save pre-processed dataset.
- `--restore_data` or `-rd`: restore pre-processed dataset from file.
- `--save_system` or `-ss`: save trained system.
- `--restore_system` or `-rs`: restore trained system from file.
- `--debug` or `-d`: use validation dataset to debug your system.
- `--interact` or `-i`: interact with your system instead of training.
- `--tensorboard` or `-tb`: enable tensorboard to monitor train performance.



## Models

In CRSLab, we unify the task description of conversational recommendation into three sub-tasks, namely recommendation (recommend user-preferred items), conversation (generate proper responses) and policy (select proper interactive action). The recommendation and conversation sub-tasks are the core of a CRS and have been studied in most of works. The policy sub-task is needed by recent works, by which the CRS can interact with users through purposeful strategy.
As the first release version, we have implemented 18 models in the four categories of CRS model, Recommendation model, Conversation model and Policy model.

|       Category       |                            Model                             |      Graph Neural Network?      |       Pre-training Model?       |
| :------------------: | :----------------------------------------------------------: | :-----------------------------: | :-----------------------------: |
|      CRS Model       | [ReDial](https://arxiv.org/abs/1812.07617)<br/>[KBRD](https://arxiv.org/abs/1908.05391)<br/>[KGSF](https://arxiv.org/abs/2007.04032)<br/>[TG-ReDial](https://arxiv.org/abs/2010.04125)<br/>[INSPIRED](https://www.aclweb.org/anthology/2020.emnlp-main.654.pdf) |       ×<br/>√<br/>√<br/>×<br/>×       |       ×<br/>×<br/>×<br/>√<br/>√       |
| Recommendation model | Popularity<br/>[GRU4Rec](https://arxiv.org/abs/1511.06939)<br/>[SASRec](https://arxiv.org/abs/1808.09781)<br/>[TextCNN](https://arxiv.org/abs/1408.5882)<br/>[R-GCN](https://arxiv.org/abs/1703.06103)<br/>[BERT](https://arxiv.org/abs/1810.04805) | ×<br/>×<br/>×<br/>×<br/>√<br/>× | ×<br/>×<br/>×<br/>×<br/>×<br/>√ |
|  Conversation model  | [HERD](https://arxiv.org/abs/1507.04808)<br/>[Transformer](https://arxiv.org/abs/1706.03762)<br/>[GPT-2](http://www.persagen.com/files/misc/radford2019language.pdf) |          ×<br/>×<br/>×          |          ×<br/>×<br/>√          |
|     Policy model     | PMI<br/>[MGCG](https://arxiv.org/abs/2005.03954)<br/>[Conv-BERT](https://arxiv.org/abs/2010.04125)<br/>[Topic-BERT](https://arxiv.org/abs/2010.04125)<br/>[Profile-BERT](https://arxiv.org/abs/2010.04125) |    ×<br/>×<br/>×<br/>×<br/>×    |    ×<br/>×<br/>√<br/>√<br/>√    |

Among them, the four CRS models integrate the recommendation model and the conversation model to improve each other, while others only specify an individual task.

For Recommendation model and Conversation model, we have respectively implemented the following commonly-used automatic evaluation metrics:

|        Category        |                           Metrics                            |
| :--------------------: | :----------------------------------------------------------: |
| Recommendation Metrics |      Hit@{1, 10, 50}, MRR@{1, 10, 50}, NDCG@{1, 10, 50}      |
|  Conversation Metrics  | PPL, BLEU-{1, 2, 3, 4}, Embedding Average/Extreme/Greedy, Distinct-{1, 2, 3, 4} |
|     Policy Metrics     |        Accuracy, Hit@{1,3,5}           |



## Datasets

We have collected and preprocessed 6 commonly-used human-annotated datasets, and each dataset was matched with proper KGs as shown below:

|                           Dataset                            | Dialogs | Utterances |   Domains    | Task Definition | Entity KG  |  Word KG   |
| :----------------------------------------------------------: | :-----: | :--------: | :----------: | :-------------: | :--------: | :--------: |
|       [ReDial](https://redialdata.github.io/website/)        | 10,006  |  182,150   |    Movie     |       --        |  DBpedia   | ConceptNet |
|      [TG-ReDial](https://github.com/RUCAIBox/TG-ReDial)      | 10,000  |  129,392   |    Movie     |   Topic Guide   | CN-DBpedia |   HowNet   |
|        [GoRecDial](https://arxiv.org/abs/1909.03922)         |  9,125  |  170,904   |    Movie     |  Action Choice  |  DBpedia   | ConceptNet |
|        [DuRecDial](https://arxiv.org/abs/2005.03954)         | 10,200  |  156,000   | Movie, Music |    Goal Plan    | CN-DBpedia |   HowNet   |
|      [INSPIRED](https://github.com/sweetpeach/Inspired)      |  1,001  |   35,811   |    Movie     | Social Strategy |  DBpedia   | ConceptNet |
| [OpenDialKG](https://github.com/facebookresearch/opendialkg) | 13,802  |   91,209   | Movie, Book  |  Path Generate  |  DBpedia   | ConceptNet |



## Performance

We have trained and test the integrated models on the TG-Redial dataset, which is split into training, validation and test sets using a ratio of 8:1:1. For each conversation, we start from the first utterance, and generate reply utterances or recommendations in turn by our model. We perform the evaluation on the three sub-tasks.

### Recommendation Task

|   Model   |    Hit@1    |   Hit@10   |   Hit@50   |    MRR@1    |   MRR@10   |   MRR@50   |   NDCG@1    |  NDCG@10   |  NDCG@50   |
| :-------: | :---------: | :--------: | :--------: | :---------: | :--------: | :--------: | :---------: | :--------: | :--------: |
|  SASRec   |  0.000446   |  0.00134   |   0.0160   |   0.000446  |  0.000576  |  0.00114   |  0.000445   |  0.00075   |  0.00380   |
|  TextCNN  |   0.00267   |   0.0103   |   0.0236   |   0.00267   |  0.00434   |  0.00493   |   0.00267   |  0.00570   |  0.00860   |
|   BERT    |   0.00722   |  0.00490   |   0.0281   |   0.00722   |   0.0106   |   0.0124   |   0.00490   |   0.0147   |   0.0239   |
|   KBRD    |   0.00401   |   0.0254   |   0.0588   |   0.00401   |  0.00891   |   0.0103   |   0.00401   |   0.0127   |   0.0198   |
|   KGSF    |   0.00535   | **0.0285** | **0.0771** |   0.00535   |   0.0114   | **0.0135** |   0.00535   | **0.0154** | **0.0259** |
| TG-ReDial | **0.00793** |   0.0251   |   0.0524   | **0.00793** | **0.0122** |   0.0134   | **0.00793** |   0.0152   |   0.0211   |


### Conversation Task

|    Model    |  BLEU@1   |  BLEU@2   |   BLEU@3   |   BLEU@4   |  Dist@1  |  Dist@2  |  Dist@3  |  Dist@4  |  Average  |  Extreme  |  Greedy   |   PPL    |
| :---------: | :-------: | :-------: | :--------: | :--------: | :------: | :------: | :------: | :------: | :-------: | :-------: | :-------: | :------: |
|    HERD     |   0.120   |  0.0141   |  0.00136   |  0.000350  |  0.181   |  0.369   |  0.847   |   1.30   |   0.697   |   0.382   |   0.639   |   472    |
| Transformer |   0.266   |  0.0440   |   0.0145   |  0.00651   |  0.324   |  0.837   |   2.02   |   3.06   |   0.879   |   0.438   |   0.680   |   30.9   |
|    GPT2     |  0.0858   |  0.0119   |  0.00377   |   0.0110   | **2.35** | **4.62** | **8.84** | **12.5** |   0.763   |   0.297   |   0.583   |   9.26   |
|    KBRD     |   0.267   |  0.0458   |   0.0134   |  0.00579   |  0.469   |   1.50   |   3.40   |   4.90   |   0.863   |   0.398   |   0.710   |   52.5   |
|    KGSF     | **0.383** | **0.115** | **0.0444** | **0.0200** |  0.340   |  0.910   |   3.50   |   6.20   | **0.888** | **0.477** | **0.767** |   50.1   |
|  TG-ReDial  |   0.125   |  0.0204   |  0.00354   |  0.000803  |  0.881   |   1.75   |   7.00   |   12.0   |   0.810   |   0.332   |   0.598   | **7.41** |


### Policy Task

|   Model    |   Hit@1   |  Hit@10   |  Hit@50   |   MRR@1   |  MRR@10   |  MRR@50   |  NDCG@1   |  NDCG@10  |  NDCG@50  |
| :--------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
|    MGCG    |   0.591   |   0.818   |   0.883   |   0.591   |   0.680   |   0.683   |   0.591   |   0.712   |   0.729   |
| Conv-BERT  |   0.597   |   0.814   |   0.881   |   0.597   |   0.684   |   0.687   |   0.597   |   0.716   |   0.731   |
| Topic-BERT |   0.598   |   0.828   |   0.885   |   0.598   |   0.690   |   0.693   |   0.598   |   0.724   |   0.737   |
| TG-ReDial  | **0.600** | **0.830** | **0.893** | **0.600** | **0.693** | **0.696** | **0.600** | **0.727** | **0.741** |

The above results were obtained from our CRSLab in preliminary experiments. However, these algorithms were implemented and tuned based on our understanding and experiences, which may not achieve their optimal performance. If you could yield a better result for some specific algorithm, please kindly let us know. We will update this table after the results are verified.

## Team

**CRSLab** was developed and maintained by [AI Box](http://aibox.ruc.edu.cn/) group in RUC.



## License

**CRSLab** uses [MIT License](./LICENSE).


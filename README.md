# Introduction

One of my colleagues asked me: "What's your most practical skill in network science?" I told him:" **community detection**"! ！

Whether you are a network science beginner, enthusiast, or expert, whether you study network data or other data networking, community detection technology will be able to accompany you at every stage of network analysis --- **from Network Data Preprocessing and Analyse, Network Visualization, Network Advanced Insight Acquisition...**

Community detection can be an organic part of your own models! The VGAER we developed provides an opportunity to combine cutting-edge GNN methods.

**Come and try!**


# VGAER

Simple and efficient -- a novel unsupervised community detection with the fusion of modularity and network structure:


<img width="754" alt="1646717854(1)" src="https://user-images.githubusercontent.com/42266769/157173553-aa740d4e-12d5-413f-86d9-91cadc7916dc.png">

#Get to Start

# VGAER Model Code Guide

## Overview

This repository contains the implementation of the VGAER (Variational Graph Auto-Encoder with Reinforcement learning) model. The model is designed to learn node representations in a graph and perform community detection. It uses a combination of graph neural networks and variational autoencoders to achieve this.

## Requirements

To run the code, You can install these packages using pip:

```bash
pip install -r requirements.txt
```

## Installation

To use the VGAER model, clone the repository and navigate to the directory:

```bash
git clone https://github.com/your-repo/vgaer-model.git
cd vgaer-model
```

## Usage

### Command-Line Arguments

The model can be configured using the following command-line arguments:

- `--model`: The type of model to use (default: `gcn_vae`)
- `--seed`: Random seed for reproducibility (default: 42)
- `--epochs`: Number of epochs to train (default: 10000)
- `--hidden1`: Number of units in the first hidden layer (default: 8)
- `--hidden2`: Number of units in the second hidden layer (default: 2)
- `--lr`: Initial learning rate (default: 0.05)
- `--dropout`: Dropout rate (default: 0.0)
- `--dataset`: Type of dataset to use (default: `cora`)
- `--cluster`: Number of communities to detect (default: 7)

### Running the Model

To run the model, use the following command:

```bash
python main.py --model gcn_vae --dataset cora --cluster 7
```

This will train the VGAER model on the Cora dataset with 7 communities.

## Code Structure

- `model.py`: Contains the implementation of the VGAER model.
- `cluster.py`: Contains community detection algorithms.
- `NMI.py`: Contains the Normalized Mutual Information (NMI) score calculation.
- `Qvalue.py`: Contains the Q value calculation for community detection.
- `main.py`: The main script that loads the dataset, trains the model, and performs community detection.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions to the project are welcome. Please open an issue or submit a pull request with your proposed changes.

## Acknowledgements

This project was inspired by various research papers and open-source projects in the field of graph neural networks and community detection. We gratefully acknowledge their contributions to the field.

# Requirement

dgl==0.8.0.post1

matplotlib==3.5.1

networkx==2.7.1

numpy==1.22.3

pandas==1.4.1

scikit_learn==1.0.2

scipy==1.8.0

seaborn==0.11.2

torch==1.11.0

# Citation

Please cite our paper if you use this code or our model in your own work:

@inproceedings{qiu2022VGAER,\
              title={VGAER: Graph Neural Network Reconstruction based Community
Detection},\
              author={Qiu, Chenyang and Huang, Zhaoci and Xu, Wenzhe and Li, Huijia},       
              booktitle={AAAI: DLG-AAAI'22},              
              year={2022}              
 }

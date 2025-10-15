# MSTVFFN 

### A multiscale spatial-temporal-variable feature fusion network for predicting multiple air pollutants ###

<font face="Times new roman" size=4>
This repo is the implementation of our manuscript entitled A multiscale spatial-temporal-variable feature fusion network for predicting multiple air pollutants. The code is based on Pytorch 1.12.1, and tested on a GeForce RTX 4090 GPU with 24GB memory.


Accurate prediction of air quality at urban monitoring stations, while accounting for the complex interactions and impacts among multiple pollutants, is crucial for enhancing urban environmental quality and public health. However, current research predominantly focuses on predicting individual pollutant indicators, without incorporating the interactions between pollutants into the modeling process, leading to limitations in prediction accuracy and capability. To address this issue, a Multiscale Spatial-Temporal-Variable Feature Fusion Network (MSTVFFN) for predicting multiple air pollutants at air quality monitoring stations is proposed. Experimental results on three urban air quality datasets showed that the proposed MSTVFFN outperformed the state-of-the-art methods in prediction various pollutants. MSTVFFN's structural framework and key modules for characterizing and fusing the multiscale features in temporal, spatial and spectral dimensions can also serve as the fundamental components of more general modeling structures for other multi-variant spatio-temporal dynamics. 

## Framework

![MSTVFFN](./Fig/MSTVFFN.png)


## Requirements
MSTVFFN uses the following dependencies
 
- Pytorch 1.12.1 and its dependencies
- Numpy and Pandas
- CUDA 11.8 or latest version

## Folder Structure
We list the code of the major modules as follows:<br>
- The main function to train/test our model: [click here](./MSTVFFN/MODEL/main.py)<br>
- The source code of our model: [click here](./MSTVFFN/MODEL/model.py)<br>
- Train and test data preporcessing are located at: [click here](./MSTVFFN/MODEL/data_preprocess.py)<br>
- Metric computations: [click here](./MSTVFFN/MODEL/utils.py)<br>

## Arguments
We introduce some major arguments of our main function here.

Training settings:
- train\_rate: rate of train set<br>
- test\_rate: rate pf test set<br>
- lag: time length of hidtorical steps<br>
- pre\_len: time length of future steps<br>
- num\_nodes: the number of stations<br>
- batch\_size: training or testing batch size<br>
- input\_dim: the feature dimension of inputs<br> 
- learning\_rate: the learning rate at the beginning<br>
- epochs: training epochs<br>
- early\_stop_patience: the patience of early stopping<br>
- device: using which GPU to train our model<br>
- seed: the random seed for experiments<br>

Model hyperparameters:<br>
- d\_model: position encoding embedding dimension<br>
- n\_heads: the number of multi-head attention<br>
- d\_k: feature dimensions of each head in multi-head attention<br>
- cheb\_k: chebyshev polynomials order<br>
- hid_dim: hidden layer dimension of Chebyshev graph convolution<br>
- dropout: dropout rate<br>


## Citation
- If you find our work useful in your research, please cite:<br>
X. Zhou, X. Liang, Q. Zhu, J. Gu, Q. Guan, A multiscale spatial–temporal-variable feature fusion network for predicting multiple air pollutants, Environment International (2025), doi: https://doi.org/10.1016/j.envint.2025.109864

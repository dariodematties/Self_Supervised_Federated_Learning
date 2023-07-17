# Federated-Learning (PyTorch)

This repository originally began as an implementation of the vanilla federated learning paper: [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629).

Experiments are produced on MNIST and CIFAR10, and both IID and non-IID sampling methods are implemented.

Since the purpose of these experiments are to probe the federated learning paradigm at a basic level, only simple models such as MLP and CNN are used.

## Requirements
Install all the packages from requirments.txt
* Python3
* Pytorch
* Torchvision

## Data
* Download train and test datasets manually or they will be automatically downloaded from torchvision datasets.
* Experiments are run on Mnist, Fashion Mnist and Cifar.
* To use your own dataset: Move your dataset to data directory and write a wrapper on pytorch dataset class.

## Running the experiments
The baseline experiment trains the model in the conventional way.

* To run the baseline experiment with MNIST on MLP using CPU:
```
python src/baseline_main.py --model=mlp --dataset=mnist --epochs=10
```
* Or to run it on GPU (eg: if gpu:0 is available):
```
python src/baseline_main.py --model=mlp --dataset=mnist --gpu=0 --epochs=10
```
-----

Federated experiment involves training a global model using many local models.

* To run the federated experiment with CIFAR on CNN (IID):
```
python src/federated_main.py --model=cnn --dataset=cifar --gpu=0 --iid=1 --epochs=10
```
* To run the same experiment under non-IID condition:
```
python src/federated_main.py --model=cnn --dataset=cifar --gpu=0 --iid=0 --epochs=10
```

You can change the default values of other parameters to simulate different conditions. Refer to the options section.

## Options
The default values for various paramters parsed to the experiment are given in ```options.py```. Details are given some of those parameters:

* ```--dataset:```  Default: 'mnist'. Options: 'mnist', 'fmnist', 'cifar'
* ```--model:```    Default: 'mlp'. Options: 'mlp', 'cnn'
* ```--gpu:```      Default: None (runs on CPU). Can also be set to the specific gpu id.
* ```--epochs:```   Number of rounds of training.
* ```--lr:```       Learning rate set to 0.01 by default.
* ```--verbose:```  Detailed log outputs. Activated by default, set to 0 to deactivate.
* ```--seed:```     Random Seed. Default set to 1.

#### Federated Parameters
* ```--iid:```      Distribution of data amongst users. Default set to IID. Set to 0 for non-IID.
* ```--num_users:```Number of users. Default is 100.
* ```--frac:```     Fraction of users to be used for federated updates. Default is 0.1.
* ```--local_ep:``` Number of local training epochs in each user. Default is 10.
* ```--local_bs:``` Batch size of local updates in each user. Default is 10.
* ```--unequal:```  Used in non-iid setting. Option to split the data amongst users equally or unequally. Default set to 0 for equal splits. Set to 1 for unequal splits.

## Results on MNIST
#### Baseline Experiment:
The experiment involves training a single model in the conventional way.

Parameters: <br />
* ```Optimizer:```    : SGD 
* ```Learning Rate:``` 0.01

```Table 1:``` Test accuracy after training for 10 epochs:

| Model | Test Acc |
| ----- | -----    |
|  MLP  |  92.71%  |
|  CNN  |  98.42%  |

----

#### Federated Experiment:
The experiment involves training a global model in the federated setting.

Federated parameters (default values):
* ```Fraction of users (C)```: 0.1 
* ```Local Batch size  (B)```: 10 
* ```Local Epochs      (E)```: 10 
* ```Optimizer            ```: SGD 
* ```Learning Rate        ```: 0.01 <br />

```Table 2:``` Test accuracy after training for 10 global epochs with:

| Model |    IID   | Non-IID (equal)|
| ----- | -----    |----            |
|  MLP  |  88.38%  |     73.49%     |
|  CNN  |  97.28%  |     75.94%     |


## Further Readings
### Papers:
* [Federated Learning: Challenges, Methods, and Future Directions](https://arxiv.org/abs/1908.07873)
* [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)
* [Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133)

### Blog Posts:
* [CMU MLD Blog Post: Federated Learning: Challenges, Methods, and Future Directions](https://blog.ml.cmu.edu/2019/11/12/federated-learning-challenges-methods-and-future-directions/)
* [Leaf: A Benchmark for Federated Settings (CMU)](https://leaf.cmu.edu/)
* [TensorFlow Federated](https://www.tensorflow.org/federated)
* [Google AI Blog Post](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html)


### Source Files
The files in the top level of the src folder consist of:
* rl_actions.py, rl_environment.py, and rl_main.py (used for some older experiments which involved situating an RL agent in a federated learning environment to make decisions about the federated training; a few changes will likely be necessary to get them working, since they were written with an old version of the utils API).
* experiments_pca_visualization.ipynb (used for generating visualizations of the PCA-reduced model parameters of client models trained on different data)
* distribution_predictor.ipynb (used for training a deep neural network on the meta-dataset to predict client label distributions)
* experiments_pca_video.ipynb (used for generating a video visualization of the evolution of PCA-reduced global model parameters across FL training)
* experiments_sharing.py and experiments_sharing_subset.py (used for experiments to assess the extent to which sharing samples across clients could improve accuracy)
* experiments_analysis.ipynb (used for conducting analysis of experiments performed using experiments_sharing.py and experiments_sharing_subset.py)

The utils folder consists of several files to support various parts of the FL training (e.g. sampling data for clients, training local models, aggregating models).

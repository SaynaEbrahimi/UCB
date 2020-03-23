# UCB in PyTorch


This is the official implementation of our paper [Uncertainty-guided Continual Learning with Bayesian Neural Networks](https://arxiv.org/abs/1906.02425), *ICLR 2020*.  

Project page:  https://sites.google.com/berkeley.edu/ucb/


## Abstract

Continual learning aims to learn new tasks without forgetting previously learned ones. This is especially challenging when one cannot access data from previous tasks and when the model has a fixed capacity. Current regularization-based continual learning algorithms need an external representation and extra computation to measure the parameters’ importance. In contrast, we propose Uncertainty- guided Continual Bayesian Neural Networks (UCB), where the learning rate adapts according to the uncertainty defined in the probability distribution of the weights in networks. Uncertainty is a natural way to identify what to remember and what to change as we continually learn, and thus mitigate catastrophic forgetting. We also show a variant of our model, which uses uncertainty for weight pruning and retains task performance after pruning by saving binary masks per tasks. We evaluate our UCB approach extensively on diverse object classification datasets with short and long sequences of tasks and report superior or on-par performance compared to existing approaches. Additionally, we show that our model does not necessarily need task information at test time, i.e. it does not presume knowledge of which task a sample belongs to.

## Authors:
[Sayna Ebrahimi](https://people.eecs.berkeley.edu/~sayna/) (UC Berkeley), [Mohamed Elhoseyni](https://sites.google.com/site/mhelhoseiny/) (Facebook AI Research, Stanford, KAIST), [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/) (UC Berkeley), [Marcus Rohrbach](http://rohrbach.vision/) (Facebook AI Research)

### Citation
If using this code, parts of it, or developments from it, please cite our paper:

```
@inproceedings{
Ebrahimi2020Uncertainty-guided,
title={Uncertainty-guided Continual Learning with Bayesian Neural Networks},
author={Sayna Ebrahimi and Mohamed Elhoseiny and Trevor Darrell and Marcus Rohrbach},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=HklUCCVKDB}
}
```

### Prerequisites:
- Linux-64
- Python 3.6
- PyTorch 1.3.1
- CPU or NVIDIA GPU + CUDA10 CuDNN7.5



### Installation
- Create a conda environment and install required packages:
```bash
conda create -n <env> python=3.6
conda activate <env>
pip install -r requirements.txt
```

- Clone this repo:
```bash
mkdir UCB
cd UCB
git clone https://github.com/SaynaEbrahimi/UCB.git
```

- The following structure is expected in the main directory:

```
./src         : main directory where all scripts are placed in
./data        : data will be automatically downloaded here
./checkpoints : results are saved in here        
```

- To run all the experiments use `src/run.py --experiment <name>` where `name` can be [`mnist2`, `mnist5`, `pmnist`, `cifar`, `mixture`].



## License
This source code is released under The MIT License found in the LICENSE file in the root directory of this source tree.


## Acknowledgements
Our code structure is inspired by [HAT](https://github.com/joansj/hat.).

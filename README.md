# Federated Dynamic Sparse Training (FedDST)

This experiment code accompanies the paper

> Sameer Bibikar, Haris Vikalo, Zhangyang Wang, and Xiaohan Chen, "Federated Dynamic Sparse Training: Computing Less, Communicating Less, Yet Learning Better," 2021.

## Dependencies
- Python 3.6 or greater
- PyTorch, torchvision
- tqdm

Run `git submodule init` followed by `git submodule update` to download the dataset code we use.

## Examples

| Experiment | Command line |
| ---------- | ------------ |
| FedAvg on CIFAR-10 | `python3 dst.py --dataset cifar10 --sparsity 0.0` |
| FedProx on CIFAR-10 (mu = 1) | `python3 dst.py --dataset cifar10 --sparsity 0.0 --prox 1` |
| FedDST on CIFAR-10 (S=0.8, alpha=0.01, R_adj=15) | `python3 dst.py --dataset cifar10 --sparsity 0.8 --readjustment-ratio 0.01 --rounds-between-readjustments 15` |
| FedDST on CIFAR-100 (S=0.5, alpha=0.01, R_adj=10) | `python3 dst.py --dataset cifar100 --sparsity 0.5 --readjustment-ratio 0.01 --distribution dirichlet --beta 0.1` |
| FedDST+FedProx on CIFAR-10 (S=0.8, alpha=0.01, R_adj=15, mu=1) | `python3 dst.py --dataset cifar10 --sparsity 0.8 --readjustment-ratio 0.01 --rounds-between-readjustments 15 --prox 1` |
| RandomMask on MNIST (S=0.8) | `python3 dst.py --dataset mnist --sparsity 0.8 --readjustment-ratio 0.0` |
| PruneFL on MNIST | `python3 prunefl.py --dataset mnist --rounds-between-readjustments 50 --initial-rounds 1000` |


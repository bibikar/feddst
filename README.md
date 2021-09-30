# feddst

## Dependencies
- Python 3.6 or greater
- PyTorch, torchvision
- tqdm

Run `git submodule init` followed by `git submodule update` to download the dataset code we use.

## Examples

| Experiment | Command line` |$
| ---------- | ------------` |$
| FedAvg on CIFAR-10 | `python3 dst.py --dataset cifar10 --sparsity 0.0` |$
| FedProx on CIFAR-10 (mu = 1) | `python3 dst.py --dataset cifar10 --sparsity 0.0 --prox 1` |$
| FedDST on CIFAR-10 (S=0.8, alpha=0.01, R_adj=15) | `python3 dst.py --dataset cifar10 --sparsity 0.8 --readjustment-ratio 0.01 --rounds-between-readjustments 15` |$
| FedDST+FedProx on CIFAR-10 (S=0.8, alpha=0.01, R_adj=15, mu=1) | `python3 dst.py --dataset cifar10 --sparsity 0.8 --readjustment-ratio 0.01 --rounds-between-readjustments 15 --prox 1` |$
| RandomMask on MNIST (S=0.8) | `python3 dst.py --dataset mnist --sparsity 0.8 --readjustment-ratio 0.0` |$
| PruneFL on MNIST | `python3 prunefl.py --dataset mnist --rounds-between-readjustments 50 --initial-rounds 1000` |$


from synflow.Utils import load, generator
from synflow.prune import *
from grasp_models import MNISTNet, CIFAR10Net, CIFAR100Net
# net = MNISTNet()
# initialize_mask(net)
# mp = generator.masked_parameters(net, False, False, False)
# print(list(mp))
def grasp(client, dataset='mnist', sparsity=0.8):
    net = {'mnist': MNISTNet, 'cifar10': CIFAR10Net, 'cifar100': CIFAR100Net}[dataset]()
    mp = generator.masked_parameters(net, False, False, False)
    print(mp)
    pruner = load.pruner('grasp')(mp)
    prune_loop(net, client.criterion, pruner, client.train_data, 'cpu', sparsity, 'exponential', 'global', 1)

    print(net.sparsity())

    return net


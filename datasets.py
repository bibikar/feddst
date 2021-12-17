import json
import numpy as np
import os
import torch
import torchvision
from tqdm import tqdm


def distribute_clients_categorical(x, p, clients=400, beta=0.1):

    unique, counts = torch.Tensor(x.targets).unique(return_counts=True)

    # Generate offsets within classes
    offsets = np.cumsum(np.broadcast_to(counts[:, np.newaxis], p.shape) * p, axis=1).astype('uint64')

    # Generate offsets for each class in the indices
    inter_class_offsets = np.cumsum(counts) - counts
    
    # Generate absolute offsets in indices for each client
    offsets = offsets + np.broadcast_to(inter_class_offsets[:, np.newaxis], offsets.shape).astype('uint64')
    offsets = np.concatenate([offsets, np.cumsum(counts)[:, np.newaxis]], axis=1).astype('uint64')

    # Use the absolute offsets as slices into the indices
    indices = []
    n_classes_by_client = []
    index_source = torch.LongTensor(np.argsort(x.targets))
    for client in range(clients):
        to_concat = []
        for noncontig_offsets in offsets[:, client:client + 2]:
            to_concat.append(index_source[slice(*noncontig_offsets)])
        indices.append(torch.cat(to_concat))
        n_classes_by_client.append(sum(1 for x in to_concat if x.numel() > 0))

    n_indices = np.array([x.numel() for x in indices])
    
    return indices, n_indices, n_classes_by_client


def distribute_clients_dirichlet(train, test, clients=400, batch_size=32, beta=0.1, rng=None):
    '''Distribute a dataset according to a Dirichlet distribution.
    '''

    rng = np.random.default_rng(rng)

    unique = torch.Tensor(train.targets).unique()

    # Generate Dirichlet samples
    alpha = np.ones(clients) * beta
    p = rng.dirichlet(alpha, size=len(unique))

    # Get indices for train and test sets
    train_idx, _, __ = distribute_clients_categorical(train, p, clients=clients, beta=beta)
    test_idx, _, __ = distribute_clients_categorical(test, p, clients=clients, beta=beta)

    return train_idx, test_idx


def distribute_iid(train, test, clients=400, samples_per_client=40, batch_size=32, rng=None):
    '''Distribute a dataset in an iid fashion, i.e. shuffle the data and then
    partition it.'''

    rng = np.random.default_rng(rng)

    train_idx = np.arange(len(train.targets))
    rng.shuffle(train_idx)
    train_idx = train_idx[:clients*samples_per_client]
    train_idx = train_idx.reshape((clients, samples_per_client))

    test_idx = np.arange(len(test.targets))
    rng.shuffle(test_idx)
    test_idx = test_idx.reshape((clients, int(len(test_idx) / clients)))

    return train_idx, test_idx


def get_mnist_or_cifar10(dataset='mnist', mode='dirichlet', path=None, clients=400,
                         classes=2, samples=20, batch_size=32, beta=0.1,
                         unbalance_rate=1.0, rng=None, **kwargs):
    '''Sample a FL dataset from MNIST, as in the LotteryFL paper.

    Parameters:
    dataset : str
        either mnist or cifar10
    path : str
        currently unused
    clients : int
        number of clients among which the dataset should be distributed
    classes : int
        number of classes each client gets in its training set
    samples : int
        number of samples per class each client gets in its training set
    batch_size : int
        batch size to use for the DataLoaders
    unbalance_rate : float
        how different the number of samples in the second class can be
        from the first, e.g. specifying samples=20, unbalance_rate=0.5 means
        the second class can be anywhere from 10-40 samples.
    rng : numpy random generator
        the RNG to use to shuffle samples.
        if None, we will grab np.random.default_rng().
        

    Returns: dict of client_id -> 2-tuples of DataLoaders
    '''

    if dataset not in ('mnist', 'cifar10', 'cifar100'):
        raise ValueError(f'unsupported dataset {dataset}')

    if path is None:
        path = os.path.join('..', 'data', dataset)

    rng = np.random.default_rng(rng)

    if mode in ('dirichlet', 'iid'):
        if dataset == 'mnist':
            xfrm = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])
            train = torchvision.datasets.MNIST(path, train=True, download=True, transform=xfrm)
            test = torchvision.datasets.MNIST(path, train=False, download=True, transform=xfrm)

        elif dataset == 'cifar10':
            xfrm = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            train = torchvision.datasets.CIFAR10(path, train=True, download=True, transform=xfrm)
            test = torchvision.datasets.CIFAR10(path, train=False, download=True, transform=xfrm)
        elif dataset == 'cifar100':
            xfrm = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            train = torchvision.datasets.CIFAR100(path, train=True, download=True, transform=xfrm)
            test = torchvision.datasets.CIFAR100(path, train=False, download=True, transform=xfrm)

        if mode == 'dirichlet':
            train_idx, test_idx = distribute_clients_dirichlet(train, test, clients=clients, beta=beta)
        elif mode == 'iid':
            train_idx, test_idx = distribute_iid(train, test, clients=clients, samples_per_client=samples)

    elif mode == 'lotteryfl':
        if dataset == 'mnist':
            from non_iid.dataset.mnist_noniid import get_dataset_mnist_extr_noniid as g
        elif dataset == 'cifar10':
            from non_iid.dataset.cifar10_noniid import get_dataset_cifar10_extr_noniid as g
        elif dataset == 'cifar100':
            from cifar100_noniid import get_dataset_cifar100_extr_noniid as g
        else:
            raise ValueError(f'dataset {dataset} is not supported by lotteryfl')

        train, test, train_idx, test_idx = g(clients, classes, samples, unbalance_rate)

    # Generate DataLoaders
    loaders = {}
    for i in range(clients):
        train_sampler = torch.LongTensor(train_idx[i])
        test_sampler = torch.LongTensor(test_idx[i])

        if len(train_sampler) == 0 or len(test_sampler) == 0:
            # ignore empty clients
            continue

        # shuffle
        train_sampler = rng.choice(train_sampler, size=train_sampler.shape, replace=False)
        test_sampler = rng.choice(test_sampler, size=test_sampler.shape, replace=False)

        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                                   sampler=train_sampler)
        test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                                  sampler=test_sampler)
        loaders[i] = (train_loader, test_loader)

    return loaders


def get_mnist(*args, **kwargs):
    return get_mnist_or_cifar10('mnist', *args, **kwargs)


def get_cifar10(*args, **kwargs):
    return get_mnist_or_cifar10('cifar10', *args, **kwargs)


def get_cifar100(*args, **kwargs):
    return get_mnist_or_cifar10('cifar100', *args, **kwargs)


def get_emnist(path='../leaf/data/femnist/data', min_samples=0, batch_size=32,
               val_size=0.2, **kwargs):
    '''Read the Federated EMNIST dataset, from the LEAF benchmark.
    The number of clients, classes per client, samples per class, and
    class imbalance are all provided as part of the dataset.

    Parameters:
    path : str
        dataset root directory
    batch_size : int
        batch size to use for DataLoaders
    val_size : float
        the relative proportion of test samples each client gets

    Returns: dict of client_id -> (train_loader, test_loader)
    '''

    EMNIST_SUBDIR = 'all_data'

    loaders = {}
    for fn in tqdm(os.listdir(os.path.join(path, EMNIST_SUBDIR))):
        fn = os.path.join(path, EMNIST_SUBDIR, fn)
        with open(fn) as f:
            subset = json.load(f)

        for uid in subset['users']:
            user_data = subset['user_data'][uid]
            data_x = (torch.FloatTensor(x).reshape((1, 28, 28)) for x in user_data['x'])
            data = list(zip(data_x, user_data['y']))

            # discard clients with less than min_samples of training data
            if len(data) < min_samples:
                continue

            n_train = int(len(data) * (1 - val_size))
            data_train = data[:n_train]
            data_test = data[n_train:]
            train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size)
            test_loader = torch.utils.data.DataLoader(data_test, batch_size=batch_size)

            loaders[uid] = (train_loader, test_loader)


    return loaders


def get_dataset(dataset, devices=None, **kwargs):
    '''Fetch the requested dataset, caching if needed

    Parameters:
    dataset : str
        either 'mnist' or 'emnist'
    devices : torch.device-like
        devices to cache the data on. If None, then minimal caching will be done.
    **kwargs
        passed to get_mnist or get_emnist

    Returns: dict of client_id -> (device, train_loader, test_loader)
    '''

    DATASET_LOADERS = {
        'mnist': get_mnist,
        'emnist': get_emnist,
        'cifar10': get_cifar10,
        'cifar100': get_cifar100
    }

    if dataset not in DATASET_LOADERS:
        raise ValueError(f'unknown dataset {dataset}. try one of {list(DATASET_LOADERS.keys())}')

    loaders = DATASET_LOADERS[dataset](**kwargs)

    if devices is None:
        return loaders

    new_loaders = {}
    for i, (uid, (train_loader, test_loader)) in enumerate(loaders.items()):
        device = devices[i % len(devices)]
        train_data = [(x.to(device), y.to(device)) for x, y in train_loader]
        test_data = [(x.to(device), y.to(device)) for x, y in test_loader]

        new_loaders[uid] = (device, train_data, test_data)

    return new_loaders


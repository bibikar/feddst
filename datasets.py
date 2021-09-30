import json
import numpy as np
import os
import torch
from tqdm import tqdm


def get_mnist_or_cifar10(dataset='mnist', path='../data/mnist', clients=400,
                         classes=2, samples=20, batch_size=32,
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

    if dataset == 'mnist':
        from non_iid.dataset.mnist_noniid import get_dataset_mnist_extr_noniid as g
    elif dataset == 'cifar10':
        from non_iid.dataset.cifar10_noniid import get_dataset_cifar10_extr_noniid as g
    else:
        raise ValueError('dataset must be mnist or cifar10 to use this function')

    train, test, train_idx, test_idx = g(clients, classes, samples, unbalance_rate)

    if not rng:
        rng = np.random.default_rng()

    # Generate DataLoaders
    loaders = {}
    for i in range(clients):
        train_sampler = train_idx[i].astype('int')
        test_sampler = test_idx[i].astype('int')

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
        'cifar10': get_cifar10
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


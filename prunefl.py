import torch
import torch.cuda
from torch import nn
from torch.nn import functional as F
import argparse
import gc
import itertools
import numpy as np
import os
import sys
import time
from copy import deepcopy

from tqdm import tqdm
import warnings

from datasets import get_dataset
import models
from models import all_models, needs_mask, initialize_mask

import pickle

rng = np.random.default_rng()

def device_list(x):
    if x == 'cpu':
        return [x]
    return [int(y) for y in x.split(',')]


parser = argparse.ArgumentParser()
parser.add_argument('--eta', type=float, help='learning rate', default=0.01)
parser.add_argument('--clients', type=int, help='number of clients per round', default=20)
parser.add_argument('--rounds', type=int, help='number of global rounds', default=400)
parser.add_argument('--epochs', type=int, help='number of local epochs', default=10)
parser.add_argument('--dataset', type=str, choices=('mnist', 'emnist', 'cifar10', 'cifar100'),
                    default='mnist', help='Dataset to use')
parser.add_argument('--distribution', type=str, choices=('dirichlet', 'lotteryfl', 'iid'), default='dirichlet',
                    help='how should the dataset be distributed?')
parser.add_argument('--beta', type=float, default=0.1, help='Beta parameter (unbalance rate) for Dirichlet distribution')
parser.add_argument('--total-clients', type=int, help='split the dataset between this many clients. Ignored for EMNIST.', default=400)
parser.add_argument('--min-samples', type=int, default=0, help='minimum number of samples required to allow a client to participate')
parser.add_argument('--prox', type=float, default=0, help='coefficient to proximal term (i.e. in FedProx)')

parser.add_argument('--batch-size', type=int, default=32,
                    help='local client batch size')
parser.add_argument('--l2', default=0.001, type=float, help='L2 regularization strength')
parser.add_argument('--momentum', default=0.9, type=float, help='Local client SGD momentum parameter')
parser.add_argument('--cache-test-set', default=False, action='store_true', help='Load test sets into memory')
parser.add_argument('--cache-test-set-gpu', default=False, action='store_true', help='Load test sets into GPU memory')
parser.add_argument('--test-batches', default=0, type=int, help='Number of minibatches to test on, or 0 for all of them')
parser.add_argument('--eval-every', default=10, type=int, help='Evaluate on test set every N rounds')
parser.add_argument('--device', default=0, type=device_list, help='Device to use for compute. Use "cpu" to force CPU. Otherwise, separate with commas to allow multi-GPU.')
parser.add_argument('--no-eval', default=True, action='store_false', dest='eval')
parser.add_argument('-o', '--outfile', default='output.log', type=argparse.FileType('a', encoding='ascii'))
parser.add_argument('--initial-rounds', default=5, type=int, help='number of "initial pruning" rounds for prunefl')
parser.add_argument('--rounds-between-readjustments', default=50, type=int, help='rounds between readj')
layer_times = [4.78686788e-05, 2.29976004e-05, 1.35797902e-06, 1.13535336e-06,
        1.06144932e-06]

args = parser.parse_args()
devices = [torch.device(x) for x in args.device]
args.pid = os.getpid()

def print2(*arg, **kwargs):
    print(*arg, **kwargs, file=args.outfile)
    print(*arg, **kwargs)

def dprint(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)

def print_csv_line(**kwargs):
    print2(','.join(str(x) for x in kwargs.values()))

def nan_to_num(x, nan=0, posinf=0, neginf=0):
    x = x.clone()
    x[x != x] = nan
    x[x == -float('inf')] = neginf
    x[x == float('inf')] = posinf
    return x.clone()


def evaluate_global(clients, global_model, progress=False, n_batches=0):
    with torch.no_grad():
        accuracies = {}
        sparsities = {}

        if progress:
            enumerator = tqdm(clients.items())
        else:
            enumerator = clients.items()

        for client_id, client in enumerator:
            accuracies[client_id] = client.test(model=global_model).item()
            sparsities[client_id] = client.sparsity()

    return accuracies, sparsities


def evaluate_local(clients, global_model, progress=False, n_batches=0):

    # we need to perform an update to client's weights.
    with torch.no_grad():
        accuracies = {}
        sparsities = {}

        if progress:
            enumerator = tqdm(clients.items())
        else:
            enumerator = clients.items()

        for client_id, client in enumerator:
            client.reset_weights(global_state=global_model.state_dict(), use_global_mask=True)
            accuracies[client_id] = client.test().item()
            sparsities[client_id] = client.sparsity()

    return accuracies, sparsities



# Fetch and cache the dataset
dprint('Fetching dataset...')
cache_devices = devices

if os.path.isfile(args.dataset + '.pickle'):
    with open(args.dataset + '.pickle', 'rb') as f:
        loaders = pickle.load(f)
else:
    loaders = get_dataset(args.dataset, clients=args.total_clients, mode=args.distribution,
                          beta=args.beta,
                          batch_size=args.batch_size, devices=cache_devices,
                          min_samples=args.min_samples)
    with open(args.dataset + '.pickle', 'wb') as f:
        pickle.dump(loaders, f)

class Client:

    def __init__(self, id, device, train_data, test_data, net=models.MNISTNet,
                 local_epochs=10, learning_rate=0.01, target_sparsity=0.1):
        '''Construct a new client.

        Parameters:
        id : object
            a unique identifier for this client. For EMNIST, this should be
            the actual client ID.
        train_data : iterable of tuples of (x, y)
            a DataLoader or other iterable giving us training samples.
        test_data : iterable of tuples of (x, y)
            a DataLoader or other iterable giving us test samples.
            (we will use this as the validation set.)
        local_epochs : int
            the number of local epochs to train for each round

        Returns: a new client.
        '''

        self.id = id

        self.train_data, self.test_data = train_data, test_data

        self.device = device
        self.net = net(device=self.device).to(self.device)
        initialize_mask(self.net)
        self.criterion = nn.CrossEntropyLoss()

        self.learning_rate = learning_rate
        self.reset_optimizer()

        self.local_epochs = local_epochs
        self.curr_epoch = 0

        # save the initial global params given to us by the server
        # for LTH pruning later.
        self.initial_global_params = None


    def reset_optimizer(self):
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.learning_rate, momentum=args.momentum, weight_decay=args.l2)


    def reset_weights(self, *args, **kwargs):
        return self.net.reset_weights(*args, **kwargs)


    def sparsity(self, *args, **kwargs):
        return self.net.sparsity(*args, **kwargs)


    def train_size(self):
        return sum(len(x) for x in self.train_data)


    def train(self, global_params=None, readjust=False):
        '''Train the client network for a single round.'''

        ul_cost = 0
        dl_cost = 0

        if global_params:
            # this is a FedAvg-like algorithm, where we need to reset
            # the client's weights every round
            mask_changed = self.reset_weights(global_state=global_params, use_global_mask=True)

            # Try to reset the optimizer state.
            self.reset_optimizer()

            if mask_changed:
                dl_cost += self.net.mask_size # need to receive mask

            if not self.initial_global_params:
                self.initial_global_params = initial_global_params
                # no DL cost here: we assume that these are transmitted as a random seed
            else:
                # otherwise, there is a DL cost: we need to receive all parameters masked '1' and
                # all parameters that don't have a mask (e.g. biases in this case)
                dl_cost += (1-self.net.sparsity()) * self.net.mask_size * 32 + (self.net.param_size - self.net.mask_size * 32)

        #pre_training_state = {k: v.clone() for k, v in self.net.state_dict().items()}
        for epoch in range(self.local_epochs):

            self.net.train()

            running_loss = 0.
            for inputs, labels in self.train_data:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                if args.prox > 0:
                    loss += args.prox / 2. * self.net.proximal_loss(global_params)
                loss.backward()
                self.optimizer.step()

                self.reset_weights() # applies the mask

                running_loss += loss.item()

            self.curr_epoch += 1

        if readjust:
            # we will upload stochastic gradients for prunable weights
            ul_cost += self.net.mask_size * 32

        # we only need to transmit the masked weights and all biases
        ul_cost += (1-self.net.sparsity()) * self.net.mask_size * 32 + (self.net.param_size - self.net.mask_size * 32)
        ret = dict(state=self.net.state_dict(), dl_cost=dl_cost, ul_cost=ul_cost)

        #dprint(global_params['conv1.weight_mask'][0, 0, 0], '->', self.net.state_dict()['conv1.weight_mask'][0, 0, 0])
        #dprint(global_params['conv1.weight'][0, 0, 0], '->', self.net.state_dict()['conv1.weight'][0, 0, 0])
        return ret

    def test(self, model=None, n_batches=0):
        '''Evaluate the local model on the local test set.

        model - model to evaluate, or this client's model if None
        n_batches - number of minibatches to test on, or 0 for all of them
        '''
        correct = 0.
        total = 0.

        if model is None:
            model = self.net
            _model = self.net
        else:
            _model = model.to(self.device)

        _model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.test_data):
                if i > n_batches and n_batches > 0:
                    break
                if not args.cache_test_set_gpu:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                outputs = _model(inputs)
                outputs = torch.argmax(outputs, dim=-1)
                correct += sum(labels == outputs)
                total += len(labels)

        # remove copies if needed
        if model is not _model:
            del _model

        return correct / total


# initialize clients
dprint('Initializing clients...')
clients = {}
client_ids = []

for i, (client_id, client_loaders) in tqdm(enumerate(loaders.items())):
    cl = Client(client_id, *client_loaders, net=all_models[args.dataset],
                learning_rate=args.eta, local_epochs=args.epochs)
    clients[client_id] = cl
    client_ids.append(client_id)
    torch.cuda.empty_cache()

# initialize global model
global_model = all_models[args.dataset](device='cpu')
initialize_mask(global_model)
initial_global_params = deepcopy(global_model.state_dict())

# we need to accumulate compute/DL/UL costs regardless of round number, resetting only
# when we actually report these numbers
compute_times = np.zeros(len(clients)) # time in seconds taken on client-side for round
download_cost = np.zeros(len(clients))
upload_cost = np.zeros(len(clients))

# pick a client randomly and perform some local readjustment there only.
# all clients are equally bad in this setting
initial_client = clients[rng.choice(list(clients.keys()), size=1)[0]]
last_differences = []
for r in tqdm(range(args.initial_rounds)):
    global_params = global_model.state_dict()
    readjust = (r - 1) % args.rounds_between_readjustments == 0
    train_result = initial_client.train(global_params=global_params, readjust=readjust)

    gradients = []
    for name, param in initial_client.net.named_parameters():
        if not needs_mask(name):
            continue
        gradients.append(param.grad)

    if readjust:
        diff = initial_client.net.prunefl_readjust(gradients, layer_times, prunable_params=0.3)
        last_differences.append(diff)
    global_model.load_state_dict(initial_client.net.state_dict())
    if len(last_differences) >= 5 and all(x < 0.1 for x in last_differences[-5:]):
        break
upload_cost[0] += train_result['ul_cost']


# for each round t = 1, 2, ... do
for server_round in tqdm(range(args.rounds)):

    # sample clients
    client_indices = rng.choice(list(clients.keys()), size=args.clients)

    global_params = global_model.state_dict()
    aggregated_params = {}
    aggregated_params_for_mask = {}
    aggregated_masks = {}
    aggregated_gradients = []
    # set server parameters to 0 in preparation for aggregation,
    for name, param in global_params.items():
        if name.endswith('_mask'):
            continue
        aggregated_params[name] = torch.zeros_like(param, dtype=torch.float, device='cpu')
        aggregated_params_for_mask[name] = torch.zeros_like(param, dtype=torch.float, device='cpu')
        if needs_mask(name):
            aggregated_masks[name] = torch.zeros_like(param, device='cpu')
            aggregated_gradients.append(torch.zeros_like(param, device='cpu'))

    # for each client k \in S_t in parallel do
    total_sampled = 0
    for client_id in client_indices:
        client = clients[client_id]
        i = client_ids.index(client_id)

        # Local client training.
        t0 = time.process_time()
        readjust = (server_round - 1) % args.rounds_between_readjustments == 0

        # actually perform training
        train_result = client.train(global_params=global_params, readjust=readjust)
        cl_params = train_result['state']
        download_cost[i] = train_result['dl_cost']
        upload_cost[i] = train_result['ul_cost']
            
        t1 = time.process_time()
        compute_times[i] = t1 - t0

        # add this client's params to the aggregate

        cl_weight_params = {}
        cl_mask_params = {}

        # first deduce masks for the received weights
        for name, cl_param in cl_params.items():
            if name.endswith('_orig'):
                name = name[:-5]
            elif name.endswith('_mask'):
                name = name[:-5]
                cl_mask_params[name] = cl_param.to(device='cpu', copy=True)
                continue

            cl_weight_params[name] = cl_param.to(device='cpu', copy=True)

        # at this point, we have weights and masks (possibly all-ones)
        # for this client. we will proceed by applying the mask and adding
        # the masked received weights to the aggregate, and adding the mask
        # to the aggregate as well.
        for name, cl_param in cl_weight_params.items():
            if name in cl_mask_params:
                # things like weights have masks
                cl_mask = cl_mask_params[name]
                sv_mask = global_params[name + '_mask'].to('cpu', copy=True)

                # calculate Hamming distance of masks for debugging
                if readjust:
                    dprint(f'{client.id} {name} d_h=', torch.sum(cl_mask ^ sv_mask).item())

                aggregated_params[name].add_(client.train_size() * cl_param * cl_mask)
                aggregated_params_for_mask[name].add_(client.train_size() * cl_param * cl_mask)
                aggregated_masks[name].add_(client.train_size() * cl_mask)
            else:
                # things like biases don't have masks
                aggregated_params[name].add_(client.train_size() * cl_param)

        # get gradients
        grad_i = 0
        for name, param in client.net.named_parameters():
            if not needs_mask(name):
                continue
            aggregated_gradients[grad_i].add_(param.grad.to('cpu'))
            grad_i += 1

    # divide gradients
    for g in aggregated_gradients:
        g.div_(args.clients)

    # at this point, we have the sum of client parameters
    # in aggregated_params, and the sum of masks in aggregated_masks. We
    # can take the average now by simply dividing...
    for name, param in aggregated_params.items():

        # if this parameter has no associated mask, simply take the average.
        if name not in aggregated_masks:
            aggregated_params[name] /= sum(clients[i].train_size() for i in client_indices)
            continue

        # otherwise, we are taking the weighted average w.r.t. the number of 
        # samples present on each of the clients.
        aggregated_params[name] /= aggregated_masks[name]
        aggregated_params_for_mask[name] /= aggregated_masks[name]
        aggregated_masks[name] /= aggregated_masks[name]

        # it's possible that some weights were pruned by all clients. In this
        # case, we will have divided by zero. Those values have already been
        # pruned out, so the values here are only placeholders.
        aggregated_params[name] = torch.nan_to_num(aggregated_params[name],
                                                   nan=0.0, posinf=0.0, neginf=0.0)
        aggregated_params_for_mask[name] = torch.nan_to_num(aggregated_params[name],
                                                   nan=0.0, posinf=0.0, neginf=0.0)
        aggregated_masks[name] = torch.nan_to_num(aggregated_masks[name],
                                                  nan=0.0, posinf=0.0, neginf=0.0)

    # masks are parameters too!
    for name, mask in aggregated_masks.items():
        aggregated_params[name + '_mask'] = mask
        aggregated_params_for_mask[name + '_mask'] = mask

    # reset global params to aggregated values
    global_model.load_state_dict(aggregated_params_for_mask)

    # perform readjustment using aggregated_gradients
    if readjust:
        global_model.prunefl_readjust(aggregated_gradients, layer_times, prunable_params=0.3 * 0.5**server_round)

    # evaluate performance
    torch.cuda.empty_cache()
    if server_round % args.eval_every == 0 and args.eval:
        accuracies, sparsities = evaluate_global(clients, global_model, progress=True,
                                                 n_batches=args.test_batches)

    for client_id in clients:
        i = client_ids.index(client_id)
        if server_round % args.eval_every == 0 and args.eval:
            print_csv_line(pid=args.pid,
                           dataset=args.dataset,
                           clients=args.clients,
                           total_clients=len(clients),
                           round=server_round,
                           batch_size=args.batch_size,
                           epochs=args.epochs,
                           target_sparsity=0.0,
                           pruning_rate=0.0,
                           initial_pruning_threshold='',
                           final_pruning_threshold='',
                           pruning_threshold_growth_method='',
                           pruning_method='PruneFL',
                           lth=False,
                           client_id=client_id,
                           accuracy=accuracies[client_id],
                           sparsity=sparsities[client_id],
                           compute_time=compute_times[i],
                           download_cost=download_cost[i],
                           upload_cost=upload_cost[i])

        # if we didn't send initial global params to any clients in the first round, send them now.
        # (in the real world, this could be implemented as the transmission of
        # a random seed, so the time and place for this is not a concern to us)
        if server_round == 0:
            clients[client_id].initial_global_params = initial_global_params

    if server_round % args.eval_every == 0 and args.eval:
        # clear compute, UL, DL costs
        compute_times[:] = 0
        download_cost[:] = 0
        upload_cost[:] = 0

#print2('OVERALL SUMMARY')
#print2()
#print2(f'{args.total_clients} clients, {args.clients} chosen each round')
#print2(f'E={args.epochs} local epochs per round, B={args.batch_size} mini-batch size')
#print2(f'{args.rounds} rounds of federated learning')
#print2(f'Target sparsity r_target={args.target_sparsity}, pruning rate (per round) r_p={args.pruning_rate}')
#print2(f'Accuracy threshold starts at {args.pruning_threshold} and ends at {args.final_pruning_threshold}')
#print2(f'Accuracy threshold growth method "{args.pruning_threshold_growth_method}"')
#print2(f'Pruning method: {args.pruning_method}, resetting weights: {args.reset_weights}')
#print2()
#print2(f'ACCURACY: mean={np.mean(accuracies)}, std={np.std(accuracies)}, min={np.min(accuracies)}, max={np.max(accuracies)}')
#print2(f'SPARSITY: mean={np.mean(sparsities)}, std={np.std(sparsities)}, min={np.min(sparsities)}, max={np.max(sparsities)}')
#print2()
#print2()


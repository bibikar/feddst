import sys
import numpy as np
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn.utils import prune
import prune as torch_prune
import warnings
from models import PrunableNet, initialize_mask

from synflow.Layers import layers
#############################
# Subclasses of PrunableNet #
#############################

class MNISTNet(PrunableNet):

    def __init__(self, *args, **kwargs):
        super(MNISTNet, self).__init__(*args, **kwargs)

        self.conv1 = layers.Conv2d(1, 10, 5) # "Conv 1-10"
        self.conv2 = layers.Conv2d(10, 20, 5) # "Conv 10-20"

        self.fc1 = layers.Linear(20 * 16 * 16, 50)
        self.fc2 = layers.Linear(50, 10)

        self.init_param_sizes()


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3, stride=1))
        x = F.relu(F.max_pool2d(self.conv2(x), 3, stride=1))
        x = x.view(-1, self.num_flat_features(x)) # flatten
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x


class CIFAR10Net(PrunableNet):

    def __init__(self, *args, **kwargs):
        super(CIFAR10Net, self).__init__(*args, **kwargs)

        self.conv1 = layers.Conv2d(3, 6, 5)
        self.conv2 = layers.Conv2d(6, 16, 5)

        self.fc1 = layers.Linear(16 * 20 * 20, 120)
        self.fc2 = layers.Linear(120, 84)
        self.fc3 = layers.Linear(84, 10)

        self.init_param_sizes()


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3, stride=1))
        x = F.relu(F.max_pool2d(self.conv2(x), 3, stride=1))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


class CIFAR100Net(PrunableNet):

    def __init__(self, *args, **kwargs):
        super(CIFAR100Net, self).__init__(*args, **kwargs)

        self.conv1 = layers.Conv2d(3, 6, 5)
        self.conv2 = layers.Conv2d(6, 16, 5)

        self.fc1 = layers.Linear(16 * 20 * 20, 120)
        self.fc2 = layers.Linear(120, 100)

        self.init_param_sizes()


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3, stride=1))
        x = F.relu(F.max_pool2d(self.conv2(x), 3, stride=1))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x
    

class EMNISTNet(PrunableNet):

    def __init__(self, *args, **kwargs):
        super(EMNISTNet, self).__init__(*args, **kwargs)

        self.conv1 = layers.Conv2d(1, 10, 5) # "Conv 1-10"
        self.conv2 = layers.Conv2d(10, 20, 5) # "Conv 10-20"

        self.fc1 = layers.Linear(20 * 16 * 16, 512)
        self.fc2 = layers.Linear(512, 62)

        self.init_param_sizes()


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3, stride=1))
        x = F.relu(F.max_pool2d(self.conv2(x), 3, stride=1))
        x = x.view(-1, self.num_flat_features(x)) # flatten
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x


class Conv2(PrunableNet):
    '''The EMNIST model from LEAF:
    https://github.com/TalwalkarLab/leaf/blob/master/models/femnist/cnn.py
    '''

    def __init__(self, *args, **kwargs):
        super(Conv2, self).__init__(*args, **kwargs)

        self.conv1 = layers.Conv2d(1, 32, 5, padding=2)
        self.conv2 = layers.Conv2d(32, 64, 5, padding=2)

        self.fc1 = layers.Linear(64 * 7 * 7, 2048)
        self.fc2 = layers.Linear(2048, 62)

        self.init_param_sizes()


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2, stride=2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x


all_models = {
        'mnist': MNISTNet,
        'emnist': Conv2,
        'cifar10': CIFAR10Net
}


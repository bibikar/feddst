import timeit
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

from mpl.models.base_model import BaseModel
from mpl.nn import DenseConv2d, DenseLinear
from mpl.optim import SGD
from datasets import get_dataset

class MNISTNet(BaseModel):

    def __init__(self):
        super(MNISTNet, self).__init__()

        conv1 = DenseConv2d.from_conv2d(nn.Conv2d(1, 10, 5)) # "Conv 1-10"
        conv2 = DenseConv2d.from_conv2d(nn.Conv2d(10, 20, 5)) # "Conv 10-20"

        self.features = nn.Sequential(conv1,
                                      nn.MaxPool2d(3, stride=1),
                                      nn.ReLU(inplace=True),
                                      conv2,
                                      nn.MaxPool2d(3, stride=1),
                                      nn.ReLU(inplace=True))

        fc1 = DenseLinear.from_linear(nn.Linear(20 * 16 * 16, 50))
        fc2 = DenseLinear.from_linear(nn.Linear(50, 10))

        self.classifier = nn.Sequential(fc1, nn.ReLU(inplace=True), fc2)

        self.collect_prunable_layers()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) # flatten
        x = F.softmax(self.classifier(x), dim=1)
        return x
    

class EMNISTNet(nn.Module):

    def __init__(self):
        super(EMNISTNet, self).__init__()

        self.conv1 = DenseConv2d.from_conv2d(nn.Conv2d(1, 10, 5)) # "Conv 1-10"
        self.conv2 = DenseConv2d.from_conv2d(nn.Conv2d(10, 20, 5)) # "Conv 10-20"

        self.fc1 = DenseLinear.from_linear(nn.Linear(20 * 16 * 16, 512))
        self.fc2 = DenseLinear.from_linear(nn.Linear(512, 62))

        self.init_param_sizes()


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3, stride=1))
        x = F.relu(F.max_pool2d(self.conv2(x), 3, stride=1))
        x = x.view(-1, x.size(0)) # flatten
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x


class CIFAR10Net(BaseModel):

    def __init__(self):
        super(CIFAR10Net, self).__init__()

        conv1 = DenseConv2d.from_conv2d(nn.Conv2d(3, 6, 5))
        conv2 = DenseConv2d.from_conv2d(nn.Conv2d(6, 16, 5))

        self.features = nn.Sequential(conv1,
                                      nn.MaxPool2d(3, stride=1),
                                      nn.ReLU(inplace=True),
                                      conv2,
                                      nn.MaxPool2d(3, stride=1),
                                      nn.ReLU(inplace=True))

        fc1 = DenseLinear.from_linear(nn.Linear(16 * 20 * 20, 120))
        fc2 = DenseLinear.from_linear(nn.Linear(120, 84))
        fc3 = DenseLinear.from_linear(nn.Linear(84, 10))

        self.classifier = nn.Sequential(fc1, nn.ReLU(inplace=True), fc2, nn.ReLU(inplace=True), fc3)

        self.collect_prunable_layers()


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.softmax(self.classifier(x), dim=1)
        return x


net = MNISTNet()
net.calc_num_prunable_params(display=True)
loaders = get_dataset('mnist', clients=400, batch_size=32, devices=[torch.device('cpu')], min_samples=0)

loader = loaders[0][1]


try_sparsities = [0.7, 0.8, 0.85, 0.9]
for sparsities in itertools.product(try_sparsities, repeat=4):

    for round in range(10):
        net = MNISTNet()
        net.calc_num_prunable_params(display=False)
        net.prune_by_pct(sparsities)
        net.calc_num_prunable_params(display=False)
        net = net.to_sparse()

        optimizer = SGD(net.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        t0 = timeit.default_timer()
        for e in range(10):
            for inputs, labels in loader:
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        t1 = timeit.default_timer()

        print(','.join(str(1 - x) for x in sparsities) + ',' + str(t1 - t0))


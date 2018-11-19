import torch
import torch.nn as nn
import torch.nn.functional as F

FILTER_SIZE = 64
NUM_CLASS = 19*19+1

class ResConv(nn.Module):
    def __init__(self):
        super(ResConv, self).__init__()
        self.conv1 = nn.Conv2d(FILTER_SIZE, FILTER_SIZE, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(FILTER_SIZE, FILTER_SIZE, 3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(FILTER_SIZE)
        self.bn2 = nn.BatchNorm2d(FILTER_SIZE)

    def forward(self, x):
        net = self.conv1(x)
        net = F.relu(self.bn1(net))
        net = self.conv2(net)
        net = self.bn2(net)
        net = net + x
        return F.relu(net)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(18, FILTER_SIZE, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(FILTER_SIZE)
        self.res_tower = nn.ModuleList([ResConv() for i in range(2)])

        # policy head
        self.conv_pol = nn.Conv2d(FILTER_SIZE, 2, 1)
        self.bn_pol = nn.BatchNorm2d(2)
        self.fc = nn.Linear(2*19*19, NUM_CLASS)

    def forward(self, x):
        """
        x is (batch, 18, 19, 19)
        """
        bsize = x.shape[0]
        net = self.conv1(x)
        net = F.relu(self.bn1(net))

        for layer in self.res_tower:
            net = layer(net)
        
        net = self.conv_pol(net)
        net = self.bn_pol(net)
        net = net.view(bsize, -1)
        net = self.fc(net)
        return F.log_softmax(net, dim=1)
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASS = 19*19+1

class ResConv(nn.Module):
    def __init__(self, filter=8):
        super(ResConv, self).__init__()
        FILTER_SIZE = filter
        self.conv1 = nn.Conv2d(FILTER_SIZE, FILTER_SIZE, 3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(FILTER_SIZE, FILTER_SIZE, 3, stride=1, padding=1, bias=False)

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
    def __init__(self, filters=256, layernum=2):
        super(Net, self).__init__()
        FILTER_SIZE = filters
        self.conv1 = nn.Conv2d(18, FILTER_SIZE, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(FILTER_SIZE)

        self.res_tower = nn.ModuleList([ResConv(FILTER_SIZE) for i in range(layernum)])

        # policy head
        self.conv_pol = nn.Conv2d(FILTER_SIZE, 2, 1, bias=False)
        self.bn_pol = nn.BatchNorm2d(2)
        self.fc_pol = nn.Linear(722, NUM_CLASS)

        # value head
        self.conv_val = nn.Conv2d(FILTER_SIZE, 1, 1, bias=False)
        self.bn_val = nn.BatchNorm2d(1)
        self.fc_val_1 = nn.Linear(361, 256)
        self.fc_val_2 = nn.Linear(256, 1)


    def forward(self, x):
        """
        x is (batch, 18, 19, 19)
        """
        net = self.conv1(x)
        net = F.relu(self.bn1(net))

        for layer in self.res_tower:
            net = layer(net)
        


        pol = self.conv_pol(net)
        pol = F.relu(self.bn_pol(pol))
        pol = pol.view(-1, 722)
        pol = self.fc_pol(pol)

        # value head
        val = self.conv_val(net)
        val = F.relu(self.bn_val(val))
        val = val.view(-1, 361)
        val = F.relu(self.fc_val_1(val))
        val = self.fc_val_2(val)


        return F.log_softmax(pol, dim=1), torch.tanh(val)

    def inference(self, x):
        """
        args
            game is a np matrix of shape (18, 19, 19)
        return
            a numpy array probability distribution over 362 moves
        """
        with torch.no_grad():
            output = self.forward(x)
            return output

class NetNoVal(nn.Module):
    def __init__(self, filters=256, layernum=2):
        super(NetNoVal, self).__init__()
        FILTER_SIZE = filters
        self.conv1 = nn.Conv2d(18, FILTER_SIZE, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(FILTER_SIZE)

        self.res_tower = nn.ModuleList([ResConv(FILTER_SIZE) for i in range(layernum)])

        # policy head
        self.conv_pol = nn.Conv2d(FILTER_SIZE, 2, 1, bias=False)
        self.bn_pol = nn.BatchNorm2d(2)
        self.fc_pol = nn.Linear(722, NUM_CLASS)


    def forward(self, x):
        """
        x is (batch, 18, 19, 19)
        """
        net = self.conv1(x)
        net = F.relu(self.bn1(net))

        for layer in self.res_tower:
            net = layer(net)

        pol = self.conv_pol(net)
        pol = F.relu(self.bn_pol(pol))
        pol = pol.view(-1, 722)
        pol = self.fc_pol(pol)

        return F.log_softmax(pol, dim=1), None

    def inference(self, x):
        """
        args
            game is a np matrix of shape (18, 19, 19)
        return
            a numpy array probability distribution over 362 moves
        """
        with torch.no_grad():
            output = self.forward(x)
            return output
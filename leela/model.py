import torch
import torch.nn as nn
import torch.nn.functional as F

from layer import *

NUM_CLASS = 19*19+1

###
###   This is the Residual Tower Model, including Baseline, Res 19, Res39
###   no winner prediction
###   Michael USE THIS
class Net_Baseline(nn.Module):
    def __init__(self, filters=256, layernum=2):
        super(Net_Baseline, self).__init__()
        FILTER_SIZE = filters
        self.conv1 = nn.Conv2d(18, FILTER_SIZE, 3, stride=1, padding=1, bias=True)
        nn.init.xavier_uniform_(self.conv1.weight)

        self.bn1 = nn.BatchNorm2d(FILTER_SIZE)

        self.res_tower = nn.ModuleList([ResConv(FILTER_SIZE) for i in range(layernum)])

        # policy head
        self.conv_pol = nn.Conv2d(FILTER_SIZE, 2, 1, bias=True)
        nn.init.xavier_uniform_(self.conv_pol.weight)

        self.bn_pol = nn.BatchNorm2d(2)
        self.fc_pol = nn.Linear(722, NUM_CLASS)
        nn.init.xavier_uniform_(self.fc_pol.weight)

        # value head
        self.conv_val = nn.Conv2d(FILTER_SIZE, 1, 1, bias=True)
        nn.init.xavier_uniform_(self.conv_val.weight)

        self.bn_val = nn.BatchNorm2d(1)
        self.fc_val_1 = nn.Linear(361, 256)
        nn.init.xavier_uniform_(self.fc_val_1.weight)

        self.fc_val_2 = nn.Linear(256, 1)
        nn.init.xavier_uniform_(self.fc_val_2.weight)


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

class NetV2(nn.Module):
    def __init__(self, filters=256, layernum=2):
        super(NetV2, self).__init__()
        FILTER_SIZE = filters
        self.conv1 = nn.Conv2d(18, FILTER_SIZE, 3, stride=1, padding=1, bias=False)
        nn.init.xavier_uniform_(self.conv1.weight)

        self.bn1 = nn.BatchNorm2d(FILTER_SIZE)

        self.res_tower = nn.ModuleList([ResConvAttn(FILTER_SIZE, fold=4) for i in range(layernum)])

        # policy head
        self.conv_pol = nn.Conv2d(FILTER_SIZE, 2, 1, bias=False)
        nn.init.xavier_uniform_(self.conv_pol.weight)

        self.bn_pol = nn.BatchNorm2d(2)
        self.fc_pol = nn.Linear(722, NUM_CLASS)
        nn.init.xavier_uniform_(self.fc_pol.weight)

        # value head
        self.conv_val = nn.Conv2d(FILTER_SIZE, 1, 1, bias=False)
        nn.init.xavier_uniform_(self.conv_val.weight)

        self.bn_val = nn.BatchNorm2d(1)
        self.fc_val_1 = nn.Linear(361, 256)
        nn.init.xavier_uniform_(self.fc_val_1.weight)

        self.fc_val_2 = nn.Linear(256, 1)
        nn.init.xavier_uniform_(self.fc_val_2.weight)


    def forward(self, x):
        """
        x is (batch, 18, 19, 19)
        """
        net = self.conv1(x)
        net = self.bn1(net)
        net = F.relu(net)

        for layer in self.res_tower:
            net = layer(net)

        # net = F.relu(net)

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

class NetV3(nn.Module):
    def __init__(self, filters=256):
        super(NetV3, self).__init__()
        FILTER_SIZE = filters
        self.filter_size = filters
        self.conv1 = nn.Conv2d(18, FILTER_SIZE, 3, stride=1, padding=1, bias=False)
        nn.init.xavier_uniform_(self.conv1.weight)

        self.bn1 = nn.BatchNorm2d(FILTER_SIZE)

        self.res_tower = nn.ModuleList([ResConv(FILTER_SIZE) for i in range(9)])
        self.attn_tower = nn.ModuleList([TransformerBlock(FILTER_SIZE) for _ in range(3)])
        # policy head
        self.conv_pol = nn.Conv2d(FILTER_SIZE, 2, 1, bias=False)
        nn.init.xavier_uniform_(self.conv_pol.weight)

        self.bn_pol = nn.BatchNorm2d(2)
        self.fc_pol = nn.Linear(722, NUM_CLASS)
        nn.init.xavier_uniform_(self.fc_pol.weight)

        # value head
        self.conv_val = nn.Conv2d(FILTER_SIZE, 1, 1, bias=False)
        nn.init.xavier_uniform_(self.conv_val.weight)

        self.bn_val = nn.BatchNorm2d(1)
        self.fc_val_1 = nn.Linear(361, 256)
        nn.init.xavier_uniform_(self.fc_val_1.weight)

        self.fc_val_2 = nn.Linear(256, 1)
        nn.init.xavier_uniform_(self.fc_val_2.weight)


    def forward(self, x):
        """
        x is (batch, 18, 19, 19)
        """
        net = self.conv1(x)
        net = self.bn1(net)
        net = F.relu(net)

        for layer in self.res_tower:
            net = layer(net)

        net = net.reshape((-1, self.filter_size, 361))
        net = net.permute(0, 2, 1)

        for layer in self.attn_tower:
            net = layer(net)

        net = net.permute(0, 2, 1)
        net = net.reshape((-1, self.filter_size, 19, 19))

        # net = F.relu(net)

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

class NetV4(nn.Module):
    def __init__(self, filters=256):
        super(NetV4, self).__init__()
        FILTER_SIZE = filters
        self.filter_size = filters
        self.conv1 = nn.Conv2d(18, FILTER_SIZE, 3, stride=1, padding=1, bias=False)
        nn.init.xavier_uniform_(self.conv1.weight)

        self.bn1 = nn.BatchNorm2d(FILTER_SIZE)

        self.res_tower = nn.ModuleList([ResConv(FILTER_SIZE) for i in range(4)])
        self.attn_tower = nn.ModuleList([TransformerBlock(FILTER_SIZE) for _ in range(3)])
        
        # policy head
        self.conv_pol = nn.Conv2d(FILTER_SIZE, 2, 1, bias=False)
        nn.init.xavier_uniform_(self.conv_pol.weight)

        self.bn_pol = nn.BatchNorm2d(2)
        self.fc_pol = nn.Linear(722, NUM_CLASS)
        nn.init.xavier_uniform_(self.fc_pol.weight)

        # value head
        self.conv_val = nn.Conv2d(FILTER_SIZE, 1, 1, bias=False)
        nn.init.xavier_uniform_(self.conv_val.weight)

        self.bn_val = nn.BatchNorm2d(1)
        self.fc_val_1 = nn.Linear(361, 256)
        nn.init.xavier_uniform_(self.fc_val_1.weight)

        self.fc_val_2 = nn.Linear(256, 1)
        nn.init.xavier_uniform_(self.fc_val_2.weight)


    def forward(self, x):
        """
        x is (batch, 18, 19, 19)
        """
        net = self.conv1(x)
        net = self.bn1(net)
        net = F.relu(net)

        for layer in self.res_tower:
            net = layer(net)

        pol = self.conv_pol(net)
        pol = F.relu(self.bn_pol(pol))
        pol = pol.view(-1, 722)
        pol = self.fc_pol(pol)

        net = net.reshape((-1, self.filter_size, 361))
        net = net.permute(0, 2, 1)

        for layer in self.attn_tower:
            net = layer(net)

        net = net.permute(0, 2, 1)
        net = net.reshape((-1, self.filter_size, 19, 19))

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

class NetV5(nn.Module):
    def __init__(self, filters=256):
        super(NetV5, self).__init__()
        FILTER_SIZE = filters
        self.filter_size = filters
        self.conv1 = nn.Conv2d(18, FILTER_SIZE, 3, stride=1, padding=1, bias=False)
        nn.init.xavier_uniform_(self.conv1.weight)

        self.bn1 = nn.BatchNorm2d(FILTER_SIZE)

        self.res_tower = nn.ModuleList([ResConv(FILTER_SIZE) for _ in range(3)])
        self.attn_tower = nn.ModuleList([TransformerBlock(FILTER_SIZE) for _ in range(3)])
        self.attn_2 = TransformerBlock(FILTER_SIZE)
        
        # policy head
        self.conv_pol = nn.Conv2d(FILTER_SIZE, 2, 1, bias=False)
        nn.init.xavier_uniform_(self.conv_pol.weight)

        self.bn_pol = nn.BatchNorm2d(2)
        self.fc_pol = nn.Linear(722, NUM_CLASS)
        nn.init.xavier_uniform_(self.fc_pol.weight)

        # value head
        self.conv_val = nn.Conv2d(FILTER_SIZE, 1, 1, bias=False)
        nn.init.xavier_uniform_(self.conv_val.weight)

        self.bn_val = nn.BatchNorm2d(1)
        self.fc_val_1 = nn.Linear(361, 256)
        nn.init.xavier_uniform_(self.fc_val_1.weight)

        self.fc_val_2 = nn.Linear(256, 1)
        nn.init.xavier_uniform_(self.fc_val_2.weight)


    def forward(self, x):
        """
        x is (batch, 18, 19, 19)
        """
        net = self.conv1(x)
        net = self.bn1(net)
        net = F.relu(net)

        for layer in self.res_tower:
            net = layer(net)

        

        net = net.reshape((-1, self.filter_size, 361))
        net = net.permute(0, 2, 1)

        net_attn_1 = net
        for layer in self.attn_tower:
            net_attn_1 = layer(net_attn_1)
        
        net = self.attn_2(net_attn_1)

        net_attn_1 = net_attn_1.permute(0, 2, 1)
        net_attn_1 = net_attn_1.reshape((-1, self.filter_size, 19, 19))

        # value head
        val = self.conv_val(net_attn_1)
        val = F.relu(self.bn_val(val))
        val = val.view(-1, 361)
        val = F.relu(self.fc_val_1(val))
        val = self.fc_val_2(val)

        
        net = net.permute(0, 2, 1)
        net = net.reshape((-1, self.filter_size, 19, 19))

        pol = self.conv_pol(net)
        pol = F.relu(self.bn_pol(pol))
        pol = pol.view(-1, 722)
        pol = self.fc_pol(pol)


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

class NetNoPol(nn.Module):
    def __init__(self, filters=256, layernum=2):
        super(NetNoPol, self).__init__()
        FILTER_SIZE = filters
        self.conv1 = nn.Conv2d(18, FILTER_SIZE, 3, stride=1, padding=1, bias=True)
        nn.init.xavier_uniform_(self.conv1.weight)

        self.bn1 = nn.BatchNorm2d(FILTER_SIZE)

        self.res_tower = nn.ModuleList([ResConv(FILTER_SIZE) for i in range(layernum)])


        # value head
        self.conv_val = nn.Conv2d(FILTER_SIZE, 1, 1, bias=True)
        nn.init.xavier_uniform_(self.conv_val.weight)

        self.bn_val = nn.BatchNorm2d(1)
        self.fc_val_1 = nn.Linear(361, 256)
        nn.init.xavier_uniform_(self.fc_val_1.weight)

        self.fc_val_2 = nn.Linear(256, 1)
        nn.init.xavier_uniform_(self.fc_val_2.weight)


    def forward(self, x):
        """
        x is (batch, 18, 19, 19)
        """
        net = self.conv1(x)
        net = F.relu(self.bn1(net))

        for layer in self.res_tower:
            net = layer(net)

        # value head
        val = self.conv_val(net)
        val = F.relu(self.bn_val(val))
        val = val.view(-1, 361)
        val = F.relu(self.fc_val_1(val))
        val = self.fc_val_2(val)


        return None , torch.tanh(val)

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

class Net_SimpleConv(nn.Module):
    def __init__(self, filters=256, layernum=2):
        super(Net_SimpleConv, self).__init__()
        FILTER_SIZE = filters
        self.conv1 = nn.Conv2d(18, FILTER_SIZE, 3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(FILTER_SIZE)

        self.res_tower = nn.ModuleList([ResConv(FILTER_SIZE) for i in range(layernum)])

        # policy head
        self.conv_pol = nn.Conv2d(FILTER_SIZE, 2, 1, bias=True)
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
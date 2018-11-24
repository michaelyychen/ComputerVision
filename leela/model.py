import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASS = 19*19+1

class MultiHeadAttn(nn.Module):
    def __init__(self, dmodel, dk, dv, head=8):
        super(MultiHeadAttn, self).__init__()
        self.model = dmodel
        self.dk = dk
        self.dv = dv
        self.head = head
        self.wQ = nn.Linear(dmodel, head*dk, bias=False)
        self.wK = nn.Linear(dmodel, head*dk, bias=False)
        self.wV = nn.Linear(dmodel, head*dv, bias=False)
        self.wO = nn.Linear(head*dv, dmodel, bias=False)

        nn.init.xavier_uniform_(self.wQ.weight)
        nn.init.xavier_uniform_(self.wK.weight)
        nn.init.xavier_uniform_(self.wV.weight)

    def forward(self, x):
        """
        x shape (batch_size, q_len, dmodel)
        """
        qlen = x.shape[1]
        qs = self.wQ(x) * (self.dk**-0.5)
        ks = self.wK(x)
        vs = self.wV(x)

        qs = qs.reshape(-1, qlen, self.head, self.dk)
        ks = ks.reshape(-1, qlen, self.head, self.dk)
        vs = vs.reshape(-1, qlen, self.head, self.dv)

        attn = torch.softmax(torch.matmul(qs.permute(0, 2, 1, 3), ks.permute(0,2,3,1)), dim=3)
        attn = torch.matmul(attn, vs.permute(0, 2, 1, 3)).permute(0, 2, 1, 3).reshape(-1, qlen, self.head*self.dv)
        output = self.wO(attn)
        return output

class ResConv(nn.Module):
    def __init__(self, filter=8):
        super(ResConv, self).__init__()
        FILTER_SIZE = filter
        self.conv1 = nn.Conv2d(FILTER_SIZE, FILTER_SIZE, 3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(FILTER_SIZE, FILTER_SIZE, 3, stride=1, padding=1, bias=True)

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)

        self.bn1 = nn.BatchNorm2d(FILTER_SIZE)
        self.bn2 = nn.BatchNorm2d(FILTER_SIZE)

    def forward(self, x):
        net = self.conv1(x)
        net = F.relu(self.bn1(net))
        net = self.conv2(net)
        net = self.bn2(net)
        net = net + x
        return F.relu(net)

class ResConvPreactivate(nn.Module):
    def __init__(self, filter=8):
        super(ResConvPreactivate, self).__init__()
        FILTER_SIZE = filter
        self.conv1 = nn.Conv2d(FILTER_SIZE, FILTER_SIZE, 3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(FILTER_SIZE, FILTER_SIZE, 3, stride=1, padding=1, bias=True)

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)

        self.bn1 = nn.BatchNorm2d(FILTER_SIZE)
        self.bn2 = nn.BatchNorm2d(FILTER_SIZE)

    def forward(self, x):
        net = F.relu(self.bn1(x))
        net = self.conv1(net)
        net = F.relu(self.bn2(net))
        net = self.conv2(net)
        net = net + x
        return net

class ResConvBottle(nn.Module):
    def __init__(self, filter=8, fold=4):
        super(ResConvBottle, self).__init__()
        FILTER_SIZE = filter
        self.conv1 = nn.Conv2d(FILTER_SIZE, FILTER_SIZE//fold, 1, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(FILTER_SIZE//fold, FILTER_SIZE//fold, 3,  stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(FILTER_SIZE//fold, FILTER_SIZE, 1,  stride=1, padding=0, bias=True)

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)

        self.bn1 = nn.BatchNorm2d(FILTER_SIZE//fold)
        self.bn2 = nn.BatchNorm2d(FILTER_SIZE//fold)
        self.bn3 = nn.BatchNorm2d(FILTER_SIZE)

    def forward(self, x):
        net = self.conv1(x)
        net = F.relu(self.bn1(net))

        net = self.conv2(net)
        net = F.relu(self.bn2(net))

        net = self.conv3(net)
        net = self.bn3(net)

        net = net + x
        return F.relu(net)

class ResConvAttn(nn.Module):
    def __init__(self, filter=8, fold=4):
        super(ResConvAttn, self).__init__()
        FILTER_SIZE = filter
        self.filter = filter
        self.conv1 = nn.Conv2d(FILTER_SIZE, FILTER_SIZE, 3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(FILTER_SIZE, FILTER_SIZE, 3,  stride=1, padding=1, bias=False)

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)

        self.bn1 = nn.BatchNorm2d(FILTER_SIZE)
        self.bn2 = nn.BatchNorm2d(FILTER_SIZE)

        self.attn = MultiHeadAttn(FILTER_SIZE, FILTER_SIZE//8, FILTER_SIZE//8, head=8)

        self.ln_attn = nn.LayerNorm(FILTER_SIZE)
        self.fc1 = nn.Linear(FILTER_SIZE, FILTER_SIZE*fold)
        self.fc2 = nn.Linear(FILTER_SIZE*fold, FILTER_SIZE)
        self.ln_fc = nn.LayerNorm(FILTER_SIZE)

    def forward(self, x):
        net = self.conv1(x)
        net = F.relu(self.bn1(net))

        net = self.conv2(net)
        net = self.bn2(net)

        net_conv = F.relu(net + x)

        # attn
        net_conv = net_conv.reshape(-1, self.filter, 19*19)
        net_conv = net_conv.permute(0, 2, 1)
        net = F.dropout(self.attn(net_conv), 0.1, training=self.training) + net_conv
        net_1 = self.ln_attn(net)

        # fc
        net = F.relu(self.fc1(net_1))
        net = F.dropout(self.fc2(net), 0.1, training=self.training) + net_1
        net = self.ln_fc(net)
        net = net.permute(0, 2, 1).reshape(-1, self.filter, 19, 19)
        return net

class ResConvBottleAttn(nn.Module):
    def __init__(self, filter=8, fold=4):
        super(ResConvBottleAttn, self).__init__()
        FILTER_SIZE = filter
        self.filter = filter
        self.conv1 = nn.Conv2d(FILTER_SIZE, FILTER_SIZE//fold, 1, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(FILTER_SIZE//fold, FILTER_SIZE//fold, 3,  stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(FILTER_SIZE//fold, FILTER_SIZE, 1,  stride=1, padding=0, bias=True)

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)

        self.bn1 = nn.BatchNorm2d(FILTER_SIZE//fold)
        self.bn2 = nn.BatchNorm2d(FILTER_SIZE//fold)
        self.bn3 = nn.BatchNorm2d(FILTER_SIZE)

        self.attn = MultiHeadAttn(FILTER_SIZE, FILTER_SIZE//8, FILTER_SIZE//8, head=8)

        self.ln_attn = nn.LayerNorm(FILTER_SIZE)
        self.fc1 = nn.Linear(FILTER_SIZE, FILTER_SIZE*fold)
        self.fc2 = nn.Linear(FILTER_SIZE*fold, FILTER_SIZE)
        self.ln_fc = nn.LayerNorm(FILTER_SIZE)

    def forward(self, x):
        net = self.conv1(x)
        net = F.relu(self.bn1(net))

        net = self.conv2(net)
        net = F.relu(self.bn2(net))

        net = self.conv3(net)
        net = self.bn3(net)

        net_conv = F.relu(net + x)

        # attn
        net_conv = net_conv.reshape(-1, self.filter, 19*19)
        net_conv = net_conv.permute(0, 2, 1)
        net = F.dropout(self.attn(net_conv), 0.1, training=self.training) + net_conv
        net_1 = self.ln_attn(net)

        # fc
        net = F.relu(self.fc1(net_1))
        net = F.dropout(self.fc2(net), 0.1, training=self.training) + net_1
        net = self.ln_fc(net)
        net = net.permute(0, 2, 1).reshape(-1, self.filter, 19, 19)
        return net

class Net(nn.Module):
    def __init__(self, filters=256, layernum=2):
        super(Net, self).__init__()
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

class NetNoVal(nn.Module):
    def __init__(self, filters=256, layernum=2):
        super(NetNoVal, self).__init__()
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
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class TransformerBlock(nn.Module):
    def __init__(self, filter=8, fold=4):
        super(TransformerBlock, self).__init__()
        FILTER_SIZE = filter
        self.filter = filter

        self.attn = MultiHeadAttn(FILTER_SIZE, FILTER_SIZE//8, FILTER_SIZE//8, head=8)

        self.ln_attn = nn.LayerNorm(FILTER_SIZE)
        self.fc1 = nn.Linear(FILTER_SIZE, FILTER_SIZE*fold)
        self.fc2 = nn.Linear(FILTER_SIZE*fold, FILTER_SIZE)
        self.ln_fc = nn.LayerNorm(FILTER_SIZE)

    def forward(self, x):
        # attn
        net = F.dropout(self.attn(x), 0.1, training=self.training) + x
        net_1 = self.ln_attn(net)

        # fc
        net = F.relu(self.fc1(net_1))
        net = F.dropout(self.fc2(net), 0.1, training=self.training) + net_1
        net = self.ln_fc(net)
        return net

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
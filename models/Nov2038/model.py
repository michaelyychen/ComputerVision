import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 361 # 19*19 
channel_size = 64
class BasicResBlock(nn.Module):
    def __init__(self, in_dim,out_dim, dropout = None, kernel_size=3, padding=1):
        super(BasicResBlock, self).__init__()
        self.dropout_rate = dropout
        self.is_dim_equal = (in_dim == out_dim)
        self.bn1 = nn.BatchNorm2d(in_dim)
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size,padding=padding)

        self.bn2 = nn.BatchNorm2d(out_dim)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=kernel_size,padding=padding)

        
        if(self.is_dim_equal):
            self.convShort = None
        else:
            self.convShort = nn.Conv2d(in_dim, out_dim,padding=0, kernel_size=1,bias = False)

    def forward(self, x):
        if(self.is_dim_equal):
            x_w = self.conv1(F.relu(self.bn1(x)))
        else:
            x = F.relu(self.bn1(x))
            x_w = self.conv1(x)

        if self.dropout_rate is not None:
            x_w = F.dropout2d(x_w,p = self.dropout_rate, training=self.training )
        x_w = self.conv2(F.relu(self.bn2(x_w)))
        if self.is_dim_equal:
            return x_w+x
        else:
            return x_w+self.convShort(x)

class ResBlock(nn.Module):
    def __init__(self, num_of_layers, in_dim, dropout = None, kernel_size=3, padding=1 ):
        super(ResBlock, self).__init__()
        self.layer = self.create_layer(num_of_layers, in_dim,dropout = dropout,kernel_size=kernel_size, padding=padding )
    def create_layer(self, num_of_layers, in_dim, dropout = None,kernel_size=3, padding=1):
        layers = []
        for i in range(int(num_of_layers)):
            layers.append(BasicResBlock(in_dim,in_dim,dropout = dropout,kernel_size=kernel_size, padding=padding))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class GoNet(nn.Module):
    def __init__(self, inChannel):
        super(GoNet, self).__init__()
        self.res1 = BasicResBlock(inChannel,channel_size,kernel_size=5, padding=2)

        self.block1 = ResBlock(10,channel_size,kernel_size=5, padding=2)
        self.res3 = BasicResBlock(channel_size,128,kernel_size=5, padding=2)
        self.block2 = ResBlock(12,128,kernel_size=3, padding=1)

        self.res2 =  BasicResBlock(128,1,kernel_size=3, padding=1)
 

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension\n",
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        
        x = self.res1(x)

        x = self.block1(x)
        x = self.res3(x)
        x = self.block2(x)

        x = self.res2(x)

        x = x.view(-1, self.num_flat_features(x))

        return F.log_softmax(x,dim=1)
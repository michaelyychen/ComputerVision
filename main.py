from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import pathlib
import time
from random import shuffle
from torch.autograd import Variable


# Training settings
parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
parser.add_argument('--data', type=str, default='../pickles', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model', type=str, default='None', metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")

args = parser.parse_args()

torch.manual_seed(args.seed)

### Data Initialization and Loading
from load import GoDataset

### Setting extra_features
### Default [b,w,e] takes 3 channels
# extra_features = ["rank_of_current_player",   # add 9 channels
#                   "rank_of_opponent",         # add 9 channels
#                   "isBlack",                  # add 1 channel
#                   "isWhite"                   # add 1 channel
#                   ]
extra_features = ["rank_of_current_player","isBlack"]


train_files = [args.data + "/train/" + file for file in os.listdir(args.data + "/train") if pathlib.Path(file).suffix == ".pickle"]
val_files = [args.data + "/val/" + file for file in os.listdir(args.data + "/val") if pathlib.Path(file).suffix == ".pickle"]

# Start Loading Data

val_datasets = [GoDataset(filename,extra_features = extra_features) for filename in val_files]
val_after_concat = torch.utils.data.ConcatDataset(val_datasets)
val_loader = torch.utils.data.DataLoader(
                        val_after_concat,
                        batch_size=args.batch_size,
                         shuffle=False, num_workers=1)

### Counting Input Channels
input_channel = 3
if "rank_of_current_player" in extra_features:
    input_channel +=9
if "rank_of_opponent" in extra_features:
    input_channel +=9
if "isBlack" in extra_features:
    input_channel +=1
if "isWhite" in extra_features:
    input_channel +=1

### Neural Network and Optimizer
from model import GoNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GoNet(input_channel).to(device)

if(args.model is not 'None'):
    state_dict = torch.load(args.model)
    model.load_state_dict(state_dict)

# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

def train(epoch):
    model.train()
    total_file = len(train_files)
    current = 1
    train_loss = 0
    num_of_data = 0
    print("Epoch {} starts at ".format(epoch)+ time.asctime(time.localtime(time.time())))
    shuffle(train_files)
    for filename in train_files:
        train_loader = torch.utils.data.DataLoader(
            GoDataset(filename,extra_features = extra_features),
                batch_size=args.batch_size, shuffle=True, num_workers=1)
        num_of_data +=len(train_loader.dataset)
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data).to(device), Variable(target).to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            train_loss += loss
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('[{}/{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    current,total_file,
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
        current +=1
    train_loss /=num_of_data
    return train_loss

def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        data, target = Variable(data,requires_grad =False).to(device), Variable(target).to(device)
        with torch.no_grad():
            output = model(data)
            validation_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    validation_loss /= len(val_loader.dataset)
    precent = "{:.2f}".format(100. * correct / len(val_loader.dataset))
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    return precent

def main():
    percent = "0"
    avg_training_loss = 0
    for epoch in range(1, args.epochs + 1):
        avg_training_loss = train(epoch)
        percent = validation()
        model_file = 'model_' + str(epoch) +"_" + percent + '.pth'
        torch.save(model.state_dict(), model_file)
    print('\nSaved model to ' + model_file + '. You can run `python evaluate.py ' + model_file + '` to generate the Kaggle formatted csv file')
if __name__ == '__main__':
    main()
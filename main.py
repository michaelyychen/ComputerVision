from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


# Training settings
parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
parser.add_argument('--data', type=str, default='pickles', metavar='D',
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
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model', type=str, default='None', metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--n_dataset', type=int, default=1, metavar='M',
                    help="how many datasets to load, from 1 to 16. each dataset contains 300,000,000 entries")
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

# Start Loading Data
training_data_name = [args.data + '/all_data_{0}.pickle'.format(i) for i in range(1,args.n_dataset+1)]
datasets = [GoDataset(filename,extra_features = extra_features) for filename in training_data_name]
after_concat = torch.utils.data.ConcatDataset(datasets)

train_loader = torch.utils.data.DataLoader(after_concat,
    batch_size=args.batch_size, shuffle=True, num_workers=1)

val_loader = torch.utils.data.DataLoader(
    GoDataset(args.data + '/all_data_17.pickle',
                         extra_features = extra_features),
    batch_size=args.batch_size, shuffle=False, num_workers=1)

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
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).to(device), Variable(target).to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

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
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    return validation_loss

def main():
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        validation()
        model_file = 'model_' + str(epoch) + '.pth'
        torch.save(model.state_dict(), model_file)
    print('\nSaved model to ' + model_file + '. You can run `python evaluate.py ' + model_file + '` to generate the Kaggle formatted csv file')
if __name__ == '__main__':
    main()
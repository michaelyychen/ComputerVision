from __future__ import print_function
import glob
import random
import gzip
import struct
import numpy as np
import time
import sys
import os

import argparse

import wandb

from chunkparser import ChunkParser
from model import Net, NetNoVal, NetNoPol

import torch
from torch.autograd import Variable
import torch.nn.functional as F

# Sane values are from 4096 to 64 or so.
# You need to adjust the learning rate if you change this. Should be
# a multiple of RAM_BATCH_SIZE. NB: It's rare that large batch sizes are
# actually required.
BATCH_SIZE = 512
# Number of examples in a GPU batch. Higher values are more efficient.
# The maximum depends on the amount of RAM in your GPU and the network size.
# Must be smaller than BATCH_SIZE.
RAM_BATCH_SIZE = 256

# Use a random sample input data read. This helps improve the spread of
# games in the shuffle buffer.
DOWN_SAMPLE = 1

FILTER_SIZE = 32
NUM_LAYER = 19



device = torch.device("cuda")

parser = argparse.ArgumentParser(description="PyTorch Go NN Trainer")
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--train-data', type=str, default='modern_games/train_',
                    help="prefix of training data")
parser.add_argument('--test-data', type=str, default='modern_games/test_',
                    help="prefix of testing data")
parser.add_argument('--checkpoints', type=str, default='checkpoints/',
                    help="prefix checkpoint folder")
parser.add_argument('--model', type=str, default=None, metavar='MOD',
                help='if provided, load the model')
parser.add_argument('--remote-log', action='store_true',
                help='if provided, do wandb')

args = parser.parse_args()

if args.remote_log:
    wandb.init()

    wandb.config.FILTER_SIZE = FILTER_SIZE
    wandb.config.NUM_BLOCK = NUM_LAYER
    wandb.config.BATCH_SIZE = RAM_BATCH_SIZE

def get_chunks(data_prefix):
    return glob.glob(data_prefix + "*.gz")

class FileDataSrc:
    """
        data source yielding chunkdata from chunk files.
    """
    def __init__(self, chunks):
        self.chunks = []
        self.done = chunks
    def next(self):
        if not self.chunks:
            self.chunks, self.done = self.done, self.chunks
            random.shuffle(self.chunks)
        if not self.chunks:
            return None
        while len(self.chunks):
            filename = self.chunks.pop()
            try:
                with gzip.open(filename, 'rb') as chunk_file:
                    self.done.append(filename)
                    return chunk_file.read()
            except:
                print("failed to parse {}".format(filename))

# 1 epoch = 62720 step

# from tensorboardX import SummaryWriter
# writer = SummaryWriter()

def compute_loss(prob, winner, target_prob, target_winner):
    if winner is None:
        return (-torch.sum(target_prob * prob)) / RAM_BATCH_SIZE
    if prob is None:
        return    F.mse_loss(winner, target_winner.unsqueeze(1))
    return (-torch.sum(target_prob * prob)/ RAM_BATCH_SIZE +
            F.mse_loss(winner, target_winner.unsqueeze(1))
            ) 

def train_loop(train_data, test_data, model, macro_batch=1, info_batch=100, eval_batch=8000):
    optimizer = torch.optim.SGD(model.parameters() ,lr=args.lr, momentum=0.9, weight_decay=1e-4)
    # optimizer = torch.optim.Adam(model.parameters() ,lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                    milestones=[50000, 100000, 150000, 200000, 250000], gamma=0.1)
                                    # milestones=[10000, 20000, 30000, 40000, 50000], gamma=0.1)

    step = 1
    
    correct = 0
    winner_correct = 0
    total_loss = 0
    start = time.time()
    struct_winner = str(RAM_BATCH_SIZE)+'f'
    print(model)
    print("Start Training")
    while True:
        scheduler.step()
        optimizer.zero_grad()

        model.train()
        (planes, probs, winner) = next(train_data)

        planes = np.frombuffer(planes, dtype=np.uint8).reshape((RAM_BATCH_SIZE, 18, 19, 19))
        probs = np.frombuffer(probs, dtype=np.float32).reshape((RAM_BATCH_SIZE, 362))
        winner = struct.unpack(struct_winner, winner)
        

        planes = Variable(torch.from_numpy(planes.astype(np.float32))).to(device)
        probs  = Variable(torch.from_numpy(probs)).to(device)

        winner = Variable(torch.FloatTensor(winner)).to(device)
        
        output_prob, output_val = model(planes)
        
        
        loss = compute_loss(output_prob, output_val, probs, winner)
        loss.backward()
        # writer.add_scalar('train_loss', loss.item(), step)

        total_loss += loss.data.cpu().item()
        # accuracy

        if output_prob is not None:
            pred = output_prob.data.max(1, keepdim=True)[1]
            label = probs.data.max(1, keepdim=True)[1]

            correct += pred.eq(label.view_as(pred)).cpu().sum().item()

        # accuracy of winner
        if output_val is not None:
            pred_val = output_val.data.sign()
            winner_correct += pred_val.eq(winner.view_as(pred_val)).cpu().sum().item()
        
        if step % macro_batch == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        if step % info_batch == 0:
            print('Time:{:.3f} Train Step: {} \tLoss: {:.6f} Acc(next, winner): {:.6f}% {:.6f}%'.format(
                time.time() - start,
                step, total_loss/ info_batch, correct / RAM_BATCH_SIZE/info_batch*100.0,
                winner_correct / RAM_BATCH_SIZE/info_batch*100.0))
            # writer.add_scalar('summary/train_loss', total_loss/ info_batch, step)
            # writer.add_scalar('summary/train_acc', correct / RAM_BATCH_SIZE/info_batch, step)
            # writer.add_scalar('summary/train_acc_winner', winner_correct / RAM_BATCH_SIZE/info_batch, step)
            # for name, param in model.named_parameters():
                # writer.add_histogram("variable/"+name, param.clone().cpu().data.numpy(), step)
                # writer.add_histogram("gradient/"+name, param.grad.clone().cpu().data.numpy(), step)
            for param_group in optimizer.param_groups:
                # writer.add_scalar('summary/lr', param_group['lr'], step)
                if args.remote_log:
                    wandb.log({
                        "lr": param_group['lr'],
                        "loss": total_loss/ info_batch, 
                        'acc': correct / RAM_BATCH_SIZE/info_batch,
                        'train_acc_winner': winner_correct / RAM_BATCH_SIZE/info_batch})
            total_loss = 0
            correct = 0
            winner_correct = 0
        if step % eval_batch == 0:
            print("Start Evaluating")
            with torch.no_grad():
                eval_size = 800
                model.eval()
                eval_loss = 0
                eval_correct = 0
                eval_correct_winner = 0
                for _ in range(eval_size):
                    (planes, probs, winner) = next(test_data)

                    planes = np.frombuffer(planes, dtype=np.uint8).reshape((RAM_BATCH_SIZE, 18, 19, 19))
                    probs = np.frombuffer(probs, dtype=np.float32).reshape((RAM_BATCH_SIZE, 362))

                    planes = Variable(torch.from_numpy(planes.astype(np.float32))).to(device)
                    probs  = Variable(torch.from_numpy(probs)).to(device)

                    winner = struct.unpack(struct_winner, winner)        
                    winner = Variable(torch.FloatTensor(winner)).to(device)
                    
                    
                    output_prob, output_val = model(planes)

                    loss = compute_loss(output_prob, output_val, probs, winner)

                    eval_loss += loss.data.cpu().item()
                    
                    # accuracy
                    if output_prob is not None:
                        pred = output_prob.data.max(1, keepdim=True)[1]
                        label = probs.data.max(1, keepdim=True)[1]

                        eval_correct += pred.eq(label.view_as(pred)).cpu().sum().item()
                    
                    # accuracy of winner
                    if output_val is not None:
                        pred_val = output_val.data.sign()
                        eval_correct_winner += pred_val.eq(winner.view_as(pred_val)).cpu().sum().item()

                print('Time:{:.3f} Eval Step: {} \tLoss: {:.6f} Acc(next, winner): {:.6f}% {:.6f}%'.format(
                    time.time() - start,
                    step, eval_loss/eval_size, eval_correct / RAM_BATCH_SIZE / eval_size*100.0,
                    eval_correct_winner / RAM_BATCH_SIZE / eval_size*100.0))
                
                # writer.add_scalar('summary/val_loss', eval_loss/eval_size, step)
                # writer.add_scalar('summary/val_acc', eval_correct / RAM_BATCH_SIZE/eval_size, step)
                # writer.add_scalar('summary/val_acc_winner', eval_correct_winner / RAM_BATCH_SIZE / eval_size, step)
                if args.remote_log:
                
                    wandb.log({"val_loss": eval_loss/eval_size, 
                    'val_acc': eval_correct / RAM_BATCH_SIZE/eval_size,
                    'val_acc_winner': eval_correct_winner / RAM_BATCH_SIZE / eval_size})

                model_file =  'model_' + str(step) + '.pth'
                torch.save(model.state_dict(), args.checkpoints + model_file)
                print('\nSaved model to ' + model_file + '.')
        step+=1

model = NetNoPol(FILTER_SIZE, NUM_LAYER)

def main(args):
    training = get_chunks(args.train_data)
    testing = get_chunks(args.test_data)
    training_parser = ChunkParser(FileDataSrc(training),
                            shuffle_size=1<<20, # 2.2GB of RAM.
                            sample=DOWN_SAMPLE,
                            batch_size=RAM_BATCH_SIZE).parse()
    testing_parser = ChunkParser(FileDataSrc(testing),
                            shuffle_size=1<<20, # 2.2GB of RAM.
                            sample=DOWN_SAMPLE,
                            batch_size=RAM_BATCH_SIZE).parse()
    print("Loading Training data form " + args.train_data)
    print("Loading Training data form " + args.test_data)
    if args.model is not None:
        print("Loading Checkpoints " + args.model)
        model.load_state_dict(torch.load(args.model))

    model.to(device)
    if args.remote_log:
        wandb.hook_torch(model, log='all')
    
    train_loop(training_parser, testing_parser, model, macro_batch=BATCH_SIZE/RAM_BATCH_SIZE)

if __name__=='__main__':
    try:
        main(args)
    except KeyboardInterrupt:
        print("Saving Newest Model to model_newest.pth")
        torch.save(model.state_dict(), args.checkpoints + "model_newest.pth")
        sys.exit(0)
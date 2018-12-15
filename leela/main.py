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
wandb.init()

from chunkparser import ChunkParser
from model import Net, NetNoVal

import torch
from torch.autograd import Variable
import torch.nn.functional as F

# Sane values are from 4096 to 64 or so.
# You need to adjust the learning rate if you change this. Should be
# a multiple of RAM_BATCH_SIZE. NB: It's rare that large batch sizes are
# actually required.
# BATCH_SIZE = 512
# Number of examples in a GPU batch. Higher values are more efficient.
# The maximum depends on the amount of RAM in your GPU and the network size.
# Must be smaller than BATCH_SIZE.
RAM_BATCH_SIZE = 128

# Use a random sample input data read. This helps improve the spread of
# games in the shuffle buffer.
DOWN_SAMPLE = 1

FILTER_SIZE = 32
NUM_LAYER = 39

wandb.config.FILTER_SIZE = FILTER_SIZE
wandb.config.NUM_BLOCK = NUM_LAYER
wandb.config.BATCH_SIZE = RAM_BATCH_SIZE

device = torch.device("cuda")

parser = argparse.ArgumentParser(description="PyTorch Go NN Trainer")
parser.add_argument('--train-data', type=str, default='modern_games/train_',
                    help="prefix of training data")
parser.add_argument('--test-data', type=str, default='modern_games/test_',
                    help="prefix of testing data")
parser.add_argument('--checkpoints', type=str, default='checkpoints/',
                    help="prefix checkpoint folder")
parser.add_argument('--model', type=str, default=None, metavar='MOD',
                help='if provided, load the model')

args = parser.parse_args()

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

from tensorboardX import SummaryWriter
writer = SummaryWriter()

def compute_loss(prob, winner, target_prob, target_winner):
    if winner is None:
        return (-torch.sum(target_prob * prob)) / RAM_BATCH_SIZE

    return (-torch.sum(target_prob * prob) +
            F.mse_loss(target_winner.unsqueeze(1), winner, reduction='sum')
            ) / RAM_BATCH_SIZE

def train_loop(train_data, test_data, model, info_batch=100, eval_batch=8000):
    optimizer = torch.optim.SGD(model.parameters() ,lr=1e-1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                    milestones=[200000, 400000, 600000, 700000, 800000], gamma=0.1)

    step = 1
    
    correct = 0
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
        
        # label = np.argmax(probs, axis=1)

        planes = Variable(torch.from_numpy(planes.astype(np.float32))).to(device)
        probs  = Variable(torch.from_numpy(probs)).to(device)

        winner = Variable(torch.FloatTensor(winner)).to(device)
        
        # label = Variable(torch.from_numpy(label.astype(np.int64))).to(device)
        
        output_prob, output_val = model(planes)
        
        # loss = F.kl_div(output_prob, probs)
        # loss = F.nll_loss(output_prob, label)
        loss = compute_loss(output_prob, output_val, probs, winner)
        loss.backward()
        writer.add_scalar('train_loss', loss.item(), step)

        total_loss += loss.data.cpu().item()
        # accuracy

        pred = output_prob.data.max(1, keepdim=True)[1]
        label = probs.data.max(1, keepdim=True)[1]

        correct += pred.eq(label.view_as(pred)).cpu().sum().item()
        #if step % macro_batch_size == 0:
        optimizer.step()
        # optimizer.zero_grad()
        
        if step % info_batch == 0:
            print('Time:{:.3f} Train Step: {} \tLoss: {:.6f} Acc.: {:.6f}%'.format(
                time.time() - start,
                step, total_loss/ info_batch, correct / RAM_BATCH_SIZE/info_batch*100.0))
            writer.add_scalar('summary/train_loss', total_loss/ info_batch, step)
            writer.add_scalar('summary/train_acc', correct / RAM_BATCH_SIZE/info_batch, step)
            wandb.log({"loss": total_loss/ info_batch, 'acc': correct / RAM_BATCH_SIZE/info_batch})
            for param_group in optimizer.param_groups:
                writer.add_scalar('summary/lr', param_group['lr'], step)
            total_loss = 0
            correct = 0
        step+=1
        if step % eval_batch == (eval_batch-1):
            print("Start Evaluating")
            with torch.no_grad():
                eval_size = 800
                model.eval()
                eval_loss = 0
                eval_correct = 0
                for i in range(eval_size):
                    (planes, probs, winner) = next(test_data)

                    planes = np.frombuffer(planes, dtype=np.uint8).reshape((RAM_BATCH_SIZE, 18, 19, 19))
                    probs = np.frombuffer(probs, dtype=np.float32).reshape((RAM_BATCH_SIZE, 362))
                    # label = np.argmax(probs, axis=1)

                    planes = Variable(torch.from_numpy(planes.astype(np.float32))).to(device)
                    probs  = Variable(torch.from_numpy(probs)).to(device)
                    # label = Variable(torch.from_numpy(label.astype(np.int64))).to(device)
                    winner = struct.unpack(struct_winner, winner)        
                    winner = Variable(torch.FloatTensor(winner)).to(device)
                    
                    
                    output_prob, output_val = model(planes)

                    #loss = F.kl_div(output, probs)
                    loss = compute_loss(output_prob, output_val, probs, winner)

                    eval_loss += loss.data.cpu().item()
                    # accuracy

                    pred = output_prob.data.max(1, keepdim=True)[1]
                    label = probs.data.max(1, keepdim=True)[1]

                    eval_correct += pred.eq(label.view_as(pred)).cpu().sum().item()
                print('Time:{:.3f} Eval Step: {} \tLoss: {:.6f} Acc.: {:.6f}%'.format(
                    time.time() - start,
                    step, eval_loss/eval_size, eval_correct / RAM_BATCH_SIZE/eval_size*100.0))
                
                writer.add_scalar('summary/val_loss', eval_loss/eval_size, step)
                writer.add_scalar('summary/val_acc', eval_correct / RAM_BATCH_SIZE/eval_size, step)
                wandb.log({"val_loss": eval_loss/eval_size, 'val_acc': eval_correct / RAM_BATCH_SIZE/eval_size})

                model_file =  'model_' + str(step) + '.pth'
                torch.save(model.state_dict(), args.checkpoints + model_file)
                torch.save(model.state_dict(), os.path.join(wandb.run.dir, model_file))
                print('\nSaved model to ' + model_file + '.')

model = Net(FILTER_SIZE, NUM_LAYER)

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
    wandb.hook_torch(model, log='all')
    train_loop(training_parser, testing_parser, model)

if __name__=='__main__':
    try:
        main(args)
    except KeyboardInterrupt:
        print("Saving Newest Model to model_newest.pth")
        torch.save(model.state_dict(), args.checkpoints + "model_newest.pth")
        sys.exit(0)

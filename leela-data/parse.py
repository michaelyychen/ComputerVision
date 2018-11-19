import glob
import random
import gzip
import struct
import numpy as np
import time

from chunkparser import ChunkParser
from model import Net

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
RAM_BATCH_SIZE = 128

# Use a random sample input data read. This helps improve the spread of
# games in the shuffle buffer.
DOWN_SAMPLE = 16

device = torch.device("cuda")

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

# 1epoch = 62720 step

def train_loop(train_data, test_data, model, macro_batch_size=1, info_batch=100):
    optimizer = torch.optim.SGD(model.parameters() ,lr=1e-1, momentum=0.9, weight_decay=1e-4)
    optimizer.zero_grad()
    step = 1
    
    correct = 0
    total_loss = 0
    start = time.time()
    while True:
        (planes, probs, _) = next(train_data)

        planes = np.frombuffer(planes, dtype=np.uint8).reshape((RAM_BATCH_SIZE, 18, 19, 19))
        probs = np.frombuffer(probs, dtype=np.float32).reshape((RAM_BATCH_SIZE, 362))
        label = np.argmax(probs, axis=1)

        planes = Variable(torch.from_numpy(planes.astype(np.float32))).to(device)
        #probs  = Variable(torch.from_numpy(probs)).to(device)
        label = Variable(torch.from_numpy(label.astype(np.int64))).to(device)
        output = model(planes)
        #loss = F.kl_div(output, probs)
        loss = F.nll_loss(output, label)
        loss.backward()

        total_loss += loss.data.cpu().item()
        # accuracy

        pred = output.data.max(1, keepdim=True)[1]
        #label = probs.data.max(1, keepdim=True)[1]

        correct += pred.eq(label.view_as(pred)).cpu().sum().item()
        #if step % macro_batch_size == 0:
        optimizer.step()
        optimizer.zero_grad()
        
        if step % info_batch == 0:
            print('Time:{:.3f} Train Step: {} \tLoss: {:.6f} Acc.: {:.6f}%'.format(
                time.time() - start,
                step, total_loss/ info_batch, correct / RAM_BATCH_SIZE/info_batch*100.0))
            total_loss = 0
            correct = 0
        
        step+=1
def main():
    training = get_chunks('train_')
    testing = get_chunks('test_')
    training_parser = ChunkParser(FileDataSrc(training),
                            shuffle_size=1<<20, # 2.2GB of RAM.
                            sample=1,
                            batch_size=RAM_BATCH_SIZE).parse()
    testing_parser = ChunkParser(FileDataSrc(testing),
                            shuffle_size=1<<20, # 2.2GB of RAM.
                            sample=1,
                            batch_size=RAM_BATCH_SIZE).parse()
    
    model = Net().to(device)

    train_loop(training_parser, testing_parser, model, 
                macro_batch_size=BATCH_SIZE//RAM_BATCH_SIZE)


if __name__=='__main__':
    main()
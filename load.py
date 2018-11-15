import os
import numpy as np
import pickle
import random
from torch.utils.data import Dataset, DataLoader
import torch

## Data format
# { 
#  "b":[][],
#  "w":[][],
#  "e":[][],
#  "b_w_level":(int,int),
#  "isBlack": bool,
#  "next":(int,int)
# }
class GoDataset(Dataset):
    def __init__(self, pickle_file, extra_features = []):
        """
        Args:
            pickle_file (string): Path to the pickle file.
            extra_features (list of string): Features other than b,w,e.
                Could be
                    "rank_of_current_player": add 9 channels
                    "rank_of_opponent": add 9 channels
                    "isBlack": add 1 channel
                    "isWhite": add 1 channel
        """
        self.data = None
        try:
            with open(pickle_file,"rb") as f:
                self.data = pickle.load(f)
        except IOError:
            print("Error Occured")
        self.extra_features = extra_features

    def __len__(self):
        if self.data == None:
            return 0
        else:
            return len(self.data)

    def __getitem__(self, idx):
        if self.data == None:
            raise Exception('Data Is Not Loaded Completely')
        label = self.data[idx]["next"][0]*19+self.data[idx]["next"][1]
        features =[torch.from_numpy(self.data[idx]["b"].astype(np.float32)),
                   torch.from_numpy(self.data[idx]["w"].astype(np.float32)),
                   torch.from_numpy(self.data[idx]["e"].astype(np.float32))]
        
        if "rank_of_current_player" in self.extra_features:
            level = 1
            if self.data[idx]["isBlack"]:
                level = self.data[idx]["b_w_level"][0]
            else:
                level = self.data[idx]["b_w_level"][1]
            for i in range(1,10):
                if i==level:
                    features.append(torch.ones((19,19)))
                else:
                    features.append(torch.zeros((19,19)))

        if "rank_of_opponent" in self.extra_features:
            level = 1
            if self.data[idx]["isBlack"]:
                level = self.data[idx]["b_w_level"][1]
            else:
                level = self.data[idx]["b_w_level"][0]
            for i in range(1,10):
                if i==level:
                    features.append(torch.ones((19,19)))
                else:
                    features.append(torch.zeros((19,19)))
        
        if "isBlack" in self.extra_features:
            if self.data[idx]["isBlack"]:
                features.append(torch.ones((19,19)))
            else:
                features.append(torch.zeros((19,19)))
        
        if "isWhite" in self.extra_features:
            if self.data[idx]["isBlack"]:
                features.append(torch.zeros((19,19)))
            else:
                features.append(torch.ones((19,19)))
        
        return (torch.stack(features),label)
 


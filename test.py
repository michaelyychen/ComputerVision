import numpy as np
import os
import pickle

with open("all_data.pickle","rb") as f:
    data = pickle.load(f)
for d in data:
    print(d["b_w_level"])
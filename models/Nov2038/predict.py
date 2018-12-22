from __future__ import print_function
import torch
import numpy as np
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(game,extra_features,model):
    """
    Args:
        game (dict): { 
                        "b":[][],  Black position, 19x19 numpy array with dtype = bool
                        "w":[][],  White position, 19x19 numpy array with dtype = bool
                        "e":[][],  Empty Position, 19x19 numpy array with dtype = bool
                        "b_w_level":(int,int),
                        "isBlack": bool,
                    }
        extra_features (list of string): Features other than b,w,e.
            Could be
                "rank_of_current_player": add 9 channels
                "rank_of_opponent": add 9 channels
                "isBlack": add 1 channel
                "isWhite": add 1 channel
    """

    features =[torch.from_numpy(game["b"].astype(np.float32)),
                torch.from_numpy(game["w"].astype(np.float32)),
                torch.from_numpy(game["e"].astype(np.float32))]
    
    if "rank_of_current_player" in extra_features:
        level = 1
        if game["isBlack"]:
            level = game["b_w_level"][0]
        else:
            level = game["b_w_level"][1]
        for i in range(1,10):
            if i==level:
                features.append(torch.ones((19,19)))
            else:
                features.append(torch.zeros((19,19)))

    if "rank_of_opponent" in extra_features:
        level = 1
        if game["isBlack"]:
            level = game["b_w_level"][1]
        else:
            level = game["b_w_level"][0]
        for i in range(1,10):
            if i==level:
                features.append(torch.ones((19,19)))
            else:
                features.append(torch.zeros((19,19)))
    
    if "isBlack" in extra_features:
        if game["isBlack"]:
            features.append(torch.ones((19,19)))
        else:
            features.append(torch.zeros((19,19)))
    
    if "isWhite" in extra_features:
        if game["isBlack"]:
            features.append(torch.zeros((19,19)))
        else:
            features.append(torch.ones((19,19)))
    
    data = torch.stack(features)


    data = data.view(1, data.size(0), data.size(1), data.size(2))
    data = Variable(data,requires_grad =False)
    model.eval()

    with torch.no_grad():
        output = model(data.to(device))
    return output

if __name__ == "__main__":
    game = {
        "b":  np.zeros((19,19),dtype = bool),
        "w":  np.zeros((19,19),dtype = bool),
        "e":  np.ones((19,19),dtype = bool),
        "b_w_level":(9,9),
        "isBlack": False
    }
    game["b"][5][5] = True
    game["e"][5][5] = False
    extra_features = ["rank_of_current_player","isBlack"]

    input_channel = 3
    if "rank_of_current_player" in extra_features:
        input_channel +=9
    if "rank_of_opponent" in extra_features:
        input_channel +=9
    if "isBlack" in extra_features:
        input_channel +=1
    if "isWhite" in extra_features:
        input_channel +=1

    from model import GoNet
    model = GoNet(input_channel)

    model_param = "model_2_32%.pth"
    state_dict = torch.load(model_param,map_location='cpu')
    model = model.cpu()
    model.load_state_dict(state_dict)
    print(predict(game,extra_features,model))

    
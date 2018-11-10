from sgfmill import sgf
import argparse
import sys
from sgfmill import ascii_boards
from sgfmill import sgf_moves
import numpy as np
import os
import pickle


## Data format
# { 
#  "b":[][],
#  "w":[][],
#  "e":[][],
#  "b_w_level":(),
#  "isBlack": bool,
#  "next":()
# }

def ConvertSgfToTrainingData(filename, all_data):
    with open(filename, "rb") as f:
        try:
            sgf_game = sgf.Sgf_game.from_bytes(f.read())
        except ValueError:
            raise Exception("bad sgf file")
    g_size = sgf_game.get_size()
    g_rule = None
    w_level = "1d"
    b_level = "1d"
    root_node = sgf_game.get_root()
    # print("board_size ", sgf_game.get_size())
    # print("Winner ", sgf_game.get_winner())
    # Get Rule, Chinese/Japanese
    if root_node.has_property('RU'):
        g_rule = root_node.get('RU')
        # print("Rule ",root_node.get('RU'))
    # Get White D(Level) 1...9d
    if root_node.has_property('WR'):
        w_level = root_node.get('WR')
        # print("White ",root_node.get('WR'))
    # Get Black D(Level) 1...9d 
    if root_node.has_property('BR'):
        b_level = root_node.get('BR')
        # print("Black ",root_node.get('BR'))

    if(g_rule != "Chinese" or g_size != 19):
        return

    black = np.zeros((19,19),dtype = bool)
    white = np.zeros((19,19),dtype = bool)
    empty = np.ones((19,19),dtype = bool)
    level = [1,1]
    if(b_level[-1] == "d"):
        level[0] = int(b_level[:-1])
    if(b_level[-1] == "d"):
        level[1] = int(w_level[:-1])
    level = tuple(level)
    for node in sgf_game.get_main_sequence():
        who , next = node.get_move()
        if next != None:
            all_data.append({
                "b":black,
                "w":white,
                "e":empty,
                "b_w_level":level,
                "isBlack": who == "b",
                "next":next
            })
            if who == "b":
                black[next[0]][next[1]] = True
            else:
                white[next[0]][next[1]] = True
            empty[next[0]][next[1]] = False

if __name__ == "__main__":
    data_folder = "Data"
    output_filename = "all_data.pickle"
    all_data = []
    for dirs in os.listdir(data_folder):
        if(dirs == ".DS_Store"):
            continue
        print("In ",dirs)
        for file in os.listdir(data_folder + "/" + dirs):
            if(file == ".DS_Store"):
                continue
            ConvertSgfToTrainingData(data_folder+"/"+dirs+"/"+file,all_data)
    with open(output_filename, "wb") as f:
        pickle.dump( all_data, f)
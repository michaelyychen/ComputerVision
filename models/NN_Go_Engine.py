import random
import torch
import numpy as np

from sgfmill.boards import Board
from gtp import *

from models.Nov2038.predict import predict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NN_Go_Engine():
    def __init__(self):
        self.board = Board(19)
        self.previous_plane = set()
        

        from models.Nov2038.model import GoNet
        self.model = GoNet(13)
        model_param = "../models/Nov2038/model_5_38.05.pth"
        state_dict = torch.load(model_param,map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(device)

    
    def clearBoard(self):
        self.board = Board(19)

    def setBoardsize(self, dim):
        if dim != 19:
            return False
        return True
    
    def play(self, move):
        self.board.play(move.vec.row, move.vec.col, move.color.abbrev())
        return True

    def genmove(self, color):
        print(self.board.board)
        while(True):
            pred = self._get_prediction(color.abbrev()=='b')
            prob, candidate_move = torch.sort(pred, descending=True)

            # print top 10 moves

            for i in range(19*19):
                pos = candidate_move[0,i].item()
                print(pos)
                rate = prob[0,i].exp().item()
                next_vert = self._get_vertex_from_pos(pos)
                if next_vert.isPass:
                    return next_vert
                elif rate < 0.06:
                    return "PASS"
                try:
                    row = next_vert.row
                    col = next_vert.col

                    print("Try ({} {})".format(row, col))
                    if self.board.board[row][col] != None:
                        raise ValueError
                    self.board.board[row][col] = color.abbrev()
                    surrounded = self.board._find_surrounded_groups(row, col)
                    print(surrounded)
                    if surrounded:
                        print(surrounded[0])
                        if len(surrounded) == 1 and surrounded[0].colour == color.abbrev():
                            self.board.board[row][col] = None
                            raise ValueError
                    self.board.board[row][col] = None
                    if self._exist_Ko_fight(row, col, color.abbrev()):
                        continue
                    self.board.play(row, col, color.abbrev())
                    self.previous_plane.add(str(self.board.list_occupied_points()))
                    for i in range(10):
                        pos = candidate_move[0,i].item()
                        rate = prob[0,i].exp().item()
                        print(self._get_vertex_from_pos(pos), rate)
                    return next_vert
                except ValueError:
                    continue

    def _get_prediction(self, isBlack):
        print("Get NN Prediction")
        game = {
            "b":  np.zeros((19,19),dtype = bool),
            "w":  np.zeros((19,19),dtype = bool),
            "e":  np.ones((19,19),dtype = bool),
            "b_w_level":(9,9),
            "isBlack": isBlack
        }
        for i in range(19):
            for j in range(19):
                if self.board.board[i][j] == 'b':
                    game["b"][i][j] = True
                elif self.board.board[i][j] == 'w':
                    game["w"][i][j] = True
                else:
                    game["e"][i][j] = True
        extra_features = ["rank_of_current_player","isBlack"]

        return predict(game, extra_features, self.model)

    @staticmethod
    def _get_vertex_from_pos(pos):
        if pos == 19*19:
            return GTPVertex(0,0,True)
        row = pos//19
        col = pos%19
        return GTPVertex(row, col)
    
    def _exist_Ko_fight(self, row, col, colour):
        tmp_board = self.board.copy()
        tmp_board.play(row,col,colour)
        return str(tmp_board.list_occupied_points()) in self.previous_plane 
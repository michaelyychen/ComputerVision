import random
import torch
import numpy as np
from torch.autograd import Variable


from sgfmill.boards import Board
from gtp import *


device = torch.device("cpu")

class NN_Engine():
    def __init__(self):
        self.board = Board(19)            
        self.prev_move = (-1, -1)
        self.history = []
        self.previous_plane = set()
        from leela.model import Net
        self.model = Net(32, 39)
        model_param = "../leela/checkpoints/model_Nov20th_44_4.pth"
        state_dict = torch.load(model_param)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(device)

    
    def clearBoard(self):
        self.board = Board(19)

    def setBoardsize(self, dim):
        if dim != 19:
            return False
        return True
    
    def play(self, move):
        self.history.append(self.board.board.copy())
        ret = self.board.play(move.vec.row, move.vec.col, move.color.abbrev())
        self.previous_plane.add(str(self.board.list_occupied_points()))
        return ret

    def genmove(self, color):
        # print(self.board.board)
        simple_ko = None
        while(True):
            pred, winrate = self._get_prediction(color.abbrev()=='b')
            prob, candidate_move = torch.sort(pred, descending=True)
            prob = prob.exp().data.cpu().numpy()
            # print top 10 moves
            # print(prob)
            total_weight = np.sum(prob)
            for i in range(19*19):
                pos = candidate_move[0,i].item()
                # print(pos)
                dice = random.uniform(0, total_weight)
                rate = prob[0,i]
                # if dice > rate:
                    # reject
                    # total_weight -= rate
                    # continue
                print(rate)
                
                print("Current winrate: {}".format(winrate.data.cpu().item()))
                if winrate.data.cpu().item() < -0.95:
                    return "resign"
                next_vert = self._get_vertex_from_pos(pos)
                if next_vert.isPass:
                    return next_vert
                elif rate < 0.06:
                    return "PASS"
                try:
                    row = next_vert.row
                    col = next_vert.col
                    if (row, col) == self.prev_move:
                        self.board.board = self.history[-2]
                        self.history = self.history[:-1]
                        continue
                    if simple_ko is not None and row == simple_ko[0] and col == simple_ko[1]:
                        raise ValueError

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
                    simple_ko = self.play( GTPMove(color, next_vert) )
                    for i in range(10):
                        pos = candidate_move[0,i].item()
                        rate = prob[0,i]
                        print(self._get_vertex_from_pos(pos), rate)
                    self.prev_move = (row, col)
                    return next_vert
                except ValueError:
                    continue

    def _get_prediction(self, isBlack):
        print("Get NN Prediction")
        game = np.zeros((18, 19, 19), dtype=np.float32)
        # construct 18 input feature
        # get side to move stone
        idx = 0
        side_to_move = 'b' if isBlack else 'w'
        other_side = 'w' if isBlack else 'b'
        for h in range(len(self.history)-1, 0, -2):
            if idx > 7:
                break
            for i in range(19):
                for j in range(19):
                    if self.history[h][i][j] == side_to_move:
                        game[idx,i,j] = 1.0
            idx += 1

        # get other side to move stone
        idx = 8
        for h in range(len(self.history)-2, 0, -2):
            if idx > 15:
                break
            for i in range(19):
                for j in range(19):
                    if self.history[h][i][j] == other_side:
                        game[idx,i,j] = 1.0
            idx += 1
        
        if isBlack:
            game[16,:,:] = 1.0
        else:
            game[17,:,:] = 1.0
        game = Variable(torch.from_numpy(game)).to(device)
        self.model.eval()
        return self.model.inference(game.unsqueeze(0))

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
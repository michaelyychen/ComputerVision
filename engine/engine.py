from gtp import *

class Interface_Go_Engine():

    def clearBoard(self):
        raise NotImplementedError

    def setBoardsize(self, dim):
        raise NotImplementedError

    def play(self, GTPMove):
        raise NotImplementedError

    def genmove(self, GTPColor):
        raise NotImplementedError


from sgfmill.boards import Board
import random
class Random_Go_Engine():
    def __init__(self):
        self.board = Board(19)
    
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
        while(True):
            row = random.randrange(19)
            col = random.randrange(19)
            try:
                print("Try ({} {})".format(row, col))
                self.board.play(row, col, color.abbrev())
                return GTPVertex(row, col)
            except ValueError:
                continue
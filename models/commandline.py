#!/usr/bin/python3

from cmd import Cmd
from gtp import *

engine = None

class GTPShell(Cmd):
    prompt = ""
    ID = -1
    # commands should be kept consistant with the implementation in do_*
    commands = [
        "protocol_version",
        "name",
        "version",
        "known_command",
        "list_commands",
        "quit",
        "boardsize",
        "clear_board",
        "komi",
        "play",
        "genmove"
    ]

    # command implementations

    # adminstrative commands
    def do_protocol_version(self, line):
        self._success("2")

    def do_name(self, line):
        self._success("EYM Go")
    
    def do_version(self, line):
        self._success("0.0.1")

    def do_known_command(self, line):
        name = line.split()[0]
        if name in self.commands:
            res = str(GTPBoolean(True))
        else:
            res = str(GTPBoolean(False))
        self._success(res)

    def do_list_commands(self, line):
        res = "\n".join(self.commands)
        self._success(res)

    def do_quit(self, line):
        return True

    # setup commands
    def do_boardsize(self, line):
        tokens = line.split()
        if len(tokens) == 0:
            self._fail("boardsize not an integer")
        else:
            try:
                boardsize = int(tokens[0])
                if boardsize > 25:
                    self._fail("unacceptable size")
                self._success()
                print(boardsize)
            except ValueError:
                self._fail("boardsize not an integer")

    def do_clear_board(self, line):
        engine.clearBoard()
        self._success()

    def do_komi(self, line):
        tokens = line.split()
        if len(tokens) == 0:
            self._fail("komi not a float")
        else:
            try:
                komi = float(tokens[0])
                self._success()
                print(komi)
            except ValueError:
                self._fail("komi not a float")

    # core play commands
    def do_play(self, line):
        try:
            vec = GTPMove.fromString(line)
            engine.play(vec)
            self._success()
        except ValueError:
            self._fail("invalid color or coordinate")

    def do_genmove(self, line):
        c = GTPColor.fromString(line)
        ret = engine.genmove(c)
        self._success(str(ret))

    def default(self, line):
        self._fail("unknown command")
    
    # helper functions
    def _response(self, prompt, ret):
        line = prompt
        if self.ID != -1:
            line += str(self.ID)
        if len(ret) > 0:
            line += " " + ret
        print(line)
        print()
    
    def _success(self, res=None):
        if res is None:
            self._response("=", "")
        else:
            self._response("=", res)

    def _fail(self, res):
        self._response("?", res)

    def precmd(self, line):
        l = line.split()
        if len(l)> 0:
            try:
                self.ID = int(l[0])
                line = "".join(l[1:])
            except ValueError:
                self.ID = -1
        return line

    def postcmd(self, stop, line):
        self.ID=-1
        return stop
    

if __name__=='__main__' and __package__ is None:
    from sys import path
    from os.path import dirname as dir

    path.append(dir(path[0]))
    __package__ = "models"
    
    # Random engine
    #from engine import Random_Go_Engine
    #engine = Random_Go_Engine()

    # NN Engine
    from NN_Go_Engine import NN_Go_Engine
    engine = NN_Go_Engine()

    GTPShell().cmdloop()
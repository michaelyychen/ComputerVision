#!/usr/bin/python3

from cmd import Cmd

class GTPBoolean():
    def __init__(self, a):
        if type(a) is bool:
            self.val = a
        else:
            raise ValueError
    def __str__(self):
        return "true" if self.val else "false"
    
    @staticmethod
    def fromString(s):
        ret = GTPBoolean(False)
        ret.val = bool(s)
        return ret

class GTPVertex():
    '''Zero-based row col with (0,0) being at bottom-left
    '''
    def __init__(self, a, b):
        if type(a) is not int or type(b) is not int:
            raise ValueError
        self.row = int(a)
        self.col = int(b)
        if self.row >= 25 or self.row < 0 or self.col >= 25 or self.col < 0:
            raise ValueError
    def __str__(self):
        return chr(self.row + ord('A')) + str(self.col+1)
    
    @staticmethod
    def fromString(s):
        if type(s) is not str:
            raise ValueError
        c = ord(s[0].lower())
        if c == ord('i'):
            raise ValueError
        if c > ord('i'):
            c-=1
        ret = GTPVertex(c - ord('a'), int(s[1:]) - 1)
        return ret

class GTPColor():
    def __init__(self, c):
        if type(c) is not bool:
            raise ValueError
        self.color = c # False black, True white

    def __str__(self):
        return "white" if self.color else "black"
    
    @staticmethod
    def fromString(s):
        ret = GTPColor(False)
        s = s.lower()
        if s == "white" or s == "w":
            ret.color = True
        elif s == "black" or s == "b":
            ret.color = False
        else:
            raise ValueError
        return ret

class GTPMove():
    def __init__(self, c, v):
        if type(c) is not GTPColor or type(v) is not GTPVertex:
            raise ValueError
        self.color = c
        self.vec = v
    
    def __str__(self):
        return "{} {}".format(self.color, self.vec)
    
    @staticmethod
    def fromString(s):
        token = s.split()
        if len(token)!=2:
            raise ValueError
        ret = GTPMove(GTPColor.fromString(token[0]), GTPVertex.fromString(token[1]))
        return ret

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
            print(vec)
        except ValueError:
            self._fail("invalid color or coordinate")

    def do_genmove(self, line):
        c = GTPColor.fromString(line)
        print(c)
        self._success(str(GTPVertex(0,0)))

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
    

if __name__=='__main__':
    try:
        GTPShell().cmdloop()
    except Exception as e:
        exit(1)
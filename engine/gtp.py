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
    def __init__(self, a, b, isPass=False):
        if type(a) is not int or type(b) is not int:
            raise ValueError
        self.row = int(a)
        self.col = int(b)
        self.isPass = isPass
        if self.row >= 25 or self.row < 0 or self.col >= 25 or self.col < 0:
            raise ValueError
    def __str__(self):
        c = chr(self.row + ord('A')) if self.row < 8 else chr(self.row + 1 + ord('A'))
        return "pass" if self.isPass else c + str(self.col+1)
    
    @staticmethod
    def fromString(s):
        if type(s) is not str:
            raise ValueError
        if s == "pass":
            ret = GTPVertex(0,0,True)
            return ret
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
    
    def abbrev(self):
        return "w" if self.color else "b"

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
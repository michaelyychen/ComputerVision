from cmd import Cmd

class GTPShell(Cmd):
    prompt = ""
    ID = -1
    def _response(self, prompt, ret):
        line = prompt
        if self.ID != -1:
            line += str(self.ID)
        if len(ret) > 0:
            line += " " + ret
        print(line)

    def do_clearboard(self, arg):
        'clear the board'
        self._response("=", "")

    def precmd(self, line):
        l = line.split()
        try:
            self.ID = int(l[0])
            line = "".join(l[1:])
        except ValueError:
            self.ID = -1
        return line

    def postcmd(self, stop, line):
        id=-1
        print()
    def do_Quit(self, line):
        return True

    def default(self, line):
        self._response("?", "")

if __name__=='__main__':
    GTPShell().cmdloop()
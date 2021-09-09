
class StatFile:

    def __init__(self, stat_file):
        self.filepath = stat_file

    def write(self, s):
        with open(self.filepath, mode='w') as f:
            f.write(s + '\n')


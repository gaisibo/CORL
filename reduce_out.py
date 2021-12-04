import os
with open('out_seq.txt', 'r') as f:
    line = f.readline()
    while line:
        if '1/9716' not in line:
            print(line)
        else:
            print(line[-900:])
        line = f.readline()

import sys
from collections import deque

num = int(sys.argv[1])
file = sys.argv[2]

with open(file, 'r') as f:
    for line in deque(f, num):
        print(line, end = '')

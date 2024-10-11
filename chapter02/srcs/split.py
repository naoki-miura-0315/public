import sys
part_num = int(sys.argv[1])
file = sys.argv[2]

import math

with open(file) as f:
    txt_list = f.readlines()
    start = 0 
    #make a division_num 
    devision_num = math.ceil(len(txt_list) / part_num)
    for _ in range(part_num):
       print(txt_list[start:start + devision_num], \
       len(txt_list[start:start + devision_num]),\
       len(txt_list))
       start += devision_num



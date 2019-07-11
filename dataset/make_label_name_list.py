#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time
import os.path
import shutil

train_file = "cfg/train.txt"
output = "cfg/id_cname.list"

listID = []
cNAME = []
with open(train_file, 'r') as the_file:
    for cnt, line in enumerate(the_file):
        #print("Line {}: {}".format(cnt, line))
        sep = line.split('/')
        cname = sep[-2]
        sep1 = sep[-1]
        sep2 = sep1.split('.')
        (_, ID) = sep2[0].split('_')
        print(cname, ID)

        if(ID not in listID):
            listID.append(ID)
            cNAME.append(cname)

with open(output, 'a') as the_file:
    for i in range(0, len(listID)):
        the_file.write(listID[i] + ',' + cNAME[i] + "\n")

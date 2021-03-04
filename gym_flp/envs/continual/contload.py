# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 16:13:47 2020

@author: User
"""

import sys
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import pickle


qapfiles = [f for f in listdir(r'C:\Users\User\sciebo2\01_Dissertation\37_Python\MIP\miplib') if isfile(join(r'C:\Users\User\sciebo2\01_Dissertation\37_Python\MIP\miplib', f))]
MIPs={}
FlowMatrices={}
Sizes={}

cnt = 0
f=[]

for file in qapfiles:
    cnt += 1
    file_path = os.path.join(r'C:\Users\User\sciebo2\01_Dissertation\37_Python\MIP\miplib', file)
    with open(file_path, "r") as datfile:
        temp=datfile.read().splitlines()
        for i, line in enumerate(temp):
            if i == 0:
                n = int(line.strip())
                MIPs[file.split(".")[0]] = n
            elif len(line)<2:
                continue
            #elif len(f)<n:
            #    temp = np.array([line.split()],dtype=float)
            #    f.append(temp[0])

        
        F = np.asarray(f)
        FlowMatrices[file.split(".")[0]]=F[:]
        
        datfile.close()
        f = []
        print(cnt)
 
#with open(r'C:\Users\User\sciebo2\01_Dissertation\37_Python\QAP\gym-flp\flp\envs\mip.pkl', 'wb') as f:
#    pickle.dump([FlowMatrices], f)

    
""" Friedhof der Code-Schnipsel:
    
    for file in qapfiles:
    cnt += 1
    file_path = os.path.join('qaplib', file)
    data = open(file_path, "r")
    
    for i,l in enumerate(data):
        if i == 0:
            n = int(l.strip())
            QAPs[file.split(".")[0]] = n
        elif len(l)<2:
            continue
        elif len(d)<n:
            temp = np.array([l.split()],dtype=int).reshape(1,n)
            d.append(temp)
        elif i>n:
            if len(f)<n:
                temp = np.array([l.split()],dtype=int).reshape(1,n)
                f.append(temp)
"""
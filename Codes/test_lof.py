# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 11:06:08 2024

@author: yyy
"""
import numpy as np 
np.set_printoptions(precision=18)

def readtrainlog():

    file_path = 'train.log'
    var_ens=[]
    with open(file_path, 'r') as file:
        for line in file:
            var=[]
            temp_lines=line.replace("\n","").split(":")
            if (str(temp_lines[0])==" x"):
                temp_line = temp_lines[1].split()
                
                for i in range(0,len(temp_line)):
                    var.append(temp_line[i])
                var = np.array(var)
                var = np.float64(var)               
                var_ens.append(tuple(var))
    return var_ens

def readtestlog():

    file_path = 'test.log'
    var_ens=[]
    with open(file_path, 'r') as file:
        for line in file:
            var=[]
            temp_lines=line.replace("\n","").split(":")
            if (str(temp_lines[0])==" x"):
                temp_line = temp_lines[1].split()
                
                for i in range(0,len(temp_line)):
                    var.append(temp_line[i]) 
                var = np.array(var)
                var = np.float64(var)
                var_ens.append(tuple(var))
                
    return var_ens


var_ens_train = readtrainlog()
var_ens_test = readtestlog ()


instances = var_ens_train + var_ens_test


 
from lof import outliers
lof = outliers(20, instances)
 
for outlier in lof:
    print (outlier["lof"],outlier["instance"])
 


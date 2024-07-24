# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 20:33:54 2022

@author: yyy
"""

from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
np.set_printoptions(precision=15)
import mpl_toolkits.axisartist as axisartist

train_MSE=[]
with open(r"C:\Users\yyy\Desktop\工程\毕业论文\实验\BiGRU_autoEncoding\threshold.out", "r") as f:  # 打开文件
    data1 = f.readlines()       
    for i in range(0,len(data1)):
        temp=data1[i].split(":")      
        train_MSE.append(temp[1].replace("\n", "")) 
      
f.close()

train_MSE=np.array(train_MSE)
train_MSE=np.float64(train_MSE)

fig = plt.figure(figsize=(12,8))


font1 = {'family': 'Times New Roman','size': 30}

sns.set(style='ticks')
plt.rcParams['font.sans-serif'] = ['Times New Roman'] # 使用字体

#坐标轴刻度线朝内
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

ax=sns.distplot(train_MSE,bins = 15,color="green",kde=True,kde_kws={"color": "blue", "lw": 3, "label": "KDE"})

plt.axline((0.05, 0), (0.05, 1), color="r",ls="-",lw=2.5)

labelsx = ax.get_xticklabels() 
[label.set_fontname('Times New Roman') for label in labelsx]
[label.set_fontsize('24') for label in labelsx]

labelsy = ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labelsy]
[label.set_fontsize('24') for label in labelsy]


plt.xlabel("",font1)
plt.ylabel("",font1)


filepath='C:\\Users\\yyy\\Desktop\\'+'threshold.tif'
plt.savefig(filepath,dpi=300, bbox_inches = 'tight')  # 保存该图片  















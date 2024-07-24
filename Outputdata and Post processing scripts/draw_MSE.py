# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 14:08:47 2022

@author: yyy
"""
import numpy as np
import matplotlib.pyplot as plt

train_MSE=[]
validate_MSE=[]
with open(r"C:\Users\yyy\Desktop\工程\毕业论文\实验\BiGRU_autoEncoding\train.out", "r") as f:  # 打开文件
#with open(r"C:\Users\yyy\Desktop\train.out", "r") as f:  # 打开文件

    data1 = f.readlines()       
    for i in range(0,len(data1)):
        temp=data1[i].split(":")

        temp1= temp[0].replace(" ", "").split(",")
        print(temp1[1])
        if (temp1[1]=="trainloss"):
            train_MSE.append(temp[1].replace("\n", "")) 
        if (temp1[1]=="validateloss"):
            validate_MSE.append(temp[1].replace("\n", "")) 
      
f.close()

train_MSE=np.array(train_MSE)
train_MSE=np.float64(train_MSE)
validate_MSE=np.array(validate_MSE)
validate_MSE=np.float64(validate_MSE)



font = {'family': 'Times New Roman','size': 17}
font1 = {'family': 'Times New Roman','size': 20}

plt.figure(figsize=(14,8))    

plt.xlim(0,49)    
plt.ylim(0.1,0)

plt.rcParams['xtick.direction']='out'
plt.rcParams['ytick.direction']='out'

ax = plt.gca()
labelsx = ax.get_xticklabels() 
[label.set_fontname('Times New Roman') for label in labelsx]
[label.set_fontsize('17') for label in labelsx]

labelsy = ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labelsy]
[label.set_fontsize('17') for label in labelsy]

ax.invert_yaxis()

#plt.xticks([0,50,100,150,200,250,300],['','50','100','150','200','250','300'])

x_axis_data = [x for x in range(50)]
plt.plot(x_axis_data, train_MSE, '-',  alpha=0.8, linewidth=2, label='Training datasets')

x_axis_data = [x for x in range(50)]
plt.plot(x_axis_data, validate_MSE, '-',  alpha=0.8, linewidth=2, label='Validation datasets')

plt.legend(fontsize = 17,prop=font)
plt.xlabel("Epochs",font1)
plt.ylabel("Reconstruction errors",font1)

filepath='C:\\Users\\yyy\\Desktop\\'+'MSE.tif'
plt.savefig(filepath,dpi=300, bbox_inches = 'tight')  # 保存该图片    
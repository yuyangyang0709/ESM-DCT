# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
 
 
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
# Hyper-parameters
sequence_length = 101        # 变量个数  =  模式变量个数
input_size = 1             # 输入每个变量的维度  =  1  面积权重平均，垂直平均
hidden_size = 16           # 神经元训练参数矩阵的维度，在神经网络计算中，输入数据进行维度转换
num_layers = 1              # 神经网络层数
num_classes = 10            # 分类结果种类   =   全连接神经元维度    全连接神经网络最终输出的向量维度
batch_size = 1             # 批处理个数  =   每次输入ensemble的个数，数据的分块处理
num_epochs = 50              # 神经网络循环次数
learning_rate = 0.001       #神经网络参数更新时，新参数=旧参数 * 学习率 *（均值）   类似于模式变量更新，dt
dropout = 0
 
# Train Dataset
num_train_dataset = 40
train_total=[]

testdir = '/home/haida_zhangshaoqing/yuyy/BiGRU_autoEncoder/data_test/zm_4_rliq/'

for i in range(0,num_train_dataset):
    var=[]
    with open(testdir+"cam/"+str(i+1), "r") as f:
        data = f.readlines()      
        for j in range(0,len(data)):
            temp=data[j].replace('[','').replace(']','').split()
            for k in range(0,len(temp)):
                var.append(temp[k])
    f.close()
    with open(testdir+"pop/"+str(i+1), "r") as f:
        data = f.readlines()      
        for j in range(0,len(data)):
            temp=data[j].replace('[','').replace(']','').split()
            for k in range(0,len(temp)):
                var.append(temp[k])    
    train_total.append(var)
    f.close()

train_total=np.array(train_total)
train_total=train_total.astype(float) 
train_total=train_total.transpose()


#standardlization

train_min = []
train_max= []
standardized_global_mean=np.zeros(train_total.shape,dtype=np.float64)

linen1=[]
with open("/home/haida_zhangshaoqing/yuyy/BiGRU_autoEncoder/min.out", "r") as f:
    data1 = f.readlines()
    for i in range(0,len(data1)):
        temp = data1[i].replace("[","").replace("]","").replace("\n","").split()
        for j in range(0,len(temp)):
            train_min.append(temp[j])
f.close()

linen2=[]
with open("/home/haida_zhangshaoqing/yuyy/BiGRU_autoEncoder/max.out", "r") as f:
    data2 = f.readlines()
    for i in range(0,len(data2)):
        temp = data2[i].replace("[","").replace("]","").replace("\n","").split()
        for j in range(0,len(temp)):
            train_max.append(temp[j])
f.close()


train_min=np.array(train_min)
train_max=np.array(train_max)

train_min=train_min.astype(float) 
train_max=train_max.astype(float)


for var in range(sequence_length):
    for file in range(num_train_dataset):
        standardized_global_mean[var,file]=(train_total[var,file]-train_min[var])/(train_max[var]-train_min[var])
    
standardized_global_mean = standardized_global_mean.transpose()
standardized_global_mean_tensor = torch.as_tensor(standardized_global_mean, dtype=torch.float32)   #(num_train_dataset,101)


      
class LstmAutoEncoder(nn.Module):
    def __init__(self, input_layer=1, hidden_layer=16, batch_size=1, num_layers=1):
        super(LstmAutoEncoder, self).__init__()

        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.encoder_lstm = nn.GRU(self.input_layer, self.hidden_layer, batch_first=True, num_layers=self.num_layers, bidirectional=True, dropout=dropout)
        self.decoder_lstm = nn.GRU(self.hidden_layer, self.hidden_layer, batch_first=True, num_layers=self.num_layers, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(self.hidden_layer*2*self.num_layers*2, sequence_length)
        self.tanh = nn.Tanh()

    def forward(self, input_x):
        input_x = input_x.view(-1, sequence_length, input_size)  # input ( batch_size, seq_len, input_size)  (1,101,1)
        # encoder
        encoder_lstm, (hide1) = self.encoder_lstm(input_x)       # h0 (num_layers * num_directions, batch_size, hidden_size)   (4,1,32)
        embedding = hide1.transpose(0,1)                         # input (batch_size, seq_len, input_size)  (1,4,32)
 
        # decoder
        decoder_lstm, (hide2) = self.decoder_lstm(embedding)     # output(batch_size,  seq_len, hidden_size * num_directions)  (1,4,64)
        output_embedding_layer = self.hidden_layer * 2 * self.num_layers *2
        decoder_lstm1 = torch.reshape(decoder_lstm,(-1,self.hidden_layer * 2 * self.num_layers *2))
        fc = self.tanh(self.fc(decoder_lstm1))                    # 

        return fc.squeeze()



model = LstmAutoEncoder()
model = model.to(device) 
model.load_state_dict(torch.load('model.ckpt'))#加载模型
model.eval()

# Loss and optimizer
criterion = nn.MSELoss() 

threshold = 0.05

passcount = 0
    
# test
for seq_true in standardized_global_mean_tensor:
         
    seq_true = seq_true.to(device)        
    seq_pred = model(seq_true)

    loss = criterion(seq_pred, seq_true)
    
    print (loss.item())
    if (loss.item() < threshold) : 
        passcount = passcount + 1
    
passing_rate = passcount/num_train_dataset

print ("pass rate: "+ str(passing_rate))


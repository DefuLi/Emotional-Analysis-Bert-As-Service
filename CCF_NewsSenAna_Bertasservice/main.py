import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

import csv
import pandas as pd
import numpy as np
import visdom

from bert_serving.client import BertClient

# 超参数定义
EPOCHS = 30  # 训练的轮数
BATCH_SIZE = 32 # 批大小
LR = 1e-3  # 网络学习率
IN_DIM = 768  # 全连接层输入维度
OUT_DIM = 3  # 全连接层输出维度

# 网络类，全连接层
class Net(nn.Module):
    # in_dim=768, out_dim=3
    def __init__(self, in_dim, out_dim):
        super(Net, self).__init__()

        # 先放一层全连接层，后期可以多放几层
        self.fc = nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        out = self.fc(x)
        return out

# pytorch的dataset类 重写getitem,len方法
class Custom_dataset(Dataset):
    def __init__(self, dataset_list):
        self.dataset = dataset_list

    def __getitem__(self, item):
        text = self.dataset[item][1]
        label = self.dataset[item][2]

        return text, label

    def __len__(self):
        return len(self.dataset)


# 加载数据集
def load_dataset(filepath):
    dataset_list = []
    f = open(filepath, 'r', encoding='utf-8')
    r = csv.reader(f)
    for item in r:
        if r.line_num == 1:
            continue
        dataset_list.append(item)
    
    # 空元素补0
    for item in dataset_list:
        if item[1].strip() == '':
            item[1] = '0'

    return dataset_list


# 计算每个batch的准确率
def  batch_accuracy(pre, label):
    pre = pre.argmax(dim=1)
    correct = torch.eq(pre, label).sum().float().item()
    accuracy = correct / float(len(label))

    return accuracy


if __name__ == "__main__":

    # visdom可视化loss
    flag = 0
    vis = visdom.Visdom()
    vis.line([0], [0], win='loss曲线', opts=dict(title='loss曲线'))
    vis.line([0], [0], win='acc曲线', opts=dict(title='acc曲线'))

    train_dataset = load_dataset('data/Train.csv')  # 7337 * 3
    test_dataset = load_dataset('data/Test.csv')  # 7356 * 3

    train_cus = Custom_dataset(train_dataset)
    train_loader = DataLoader(dataset=train_cus, batch_size=BATCH_SIZE, shuffle=False)

    net = Net(IN_DIM, OUT_DIM)
    optimizer = optim.Adam(net.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    bertclient = BertClient()

    # 训练
    for epoch in range(EPOCHS):

        step = -1

        for text, label in train_loader:  #  该写这里了
            # tuple转list
            text = list(text)
            label = list(label)
            label = list(map(int, label))

            # 使用中文bert，生成句向量
            sen_vec = bertclient.encode(text)
            sen_vec = torch.tensor(sen_vec)
            label = torch.LongTensor(label)

            label = label.cuda()


            # 输入到网络中，反向传播
            pre = net(sen_vec).cuda()
            loss = criterion(pre, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 更新loss曲线，并计算准确率
            step = step + 1
            flag = flag + 1
            if step % 100 == 0:
                acc = batch_accuracy(pre, label)
                print('epoch:{} | batch:{} | acc:{} | loss:{}'.format(epoch, step, acc, loss.item()))

                vis.line([loss.item()], [flag], win='loss曲线', update='append')
                vis.line([acc], [flag], win='acc曲线', update='append')

    # 保存模型参数
    torch.save(net.state_dict(), 'net.pt')


    # 测试
    net.load_state_dict(torch.load('net.pt'))

    test_result = []
    for item in test_dataset:
        
        sen_vec = bertclient.encode([item[1]])
        sen_vec = torch.tensor(sen_vec)
        
        with torch.no_grad():
            pre = net(sen_vec).cuda()
            pre = pre.argmax(dim=1)
            pre = pre.item()
            test_result.append([item[0], pre])
    
    # 写入csv文件
    df = pd.DataFrame(test_result)
    df.to_csv('test_result.csv',index=False, header=['id', 'label'])






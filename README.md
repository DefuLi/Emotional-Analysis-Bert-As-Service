# 互联网新闻情感分析-基于Bert-As-Service

## 1 赛题简介
本项目基于中文Bert模型，没有对Bert模型进行微调，直接使用的Bert-As-Service库生成的各句子的768维句向量。<br>
具体赛题信息见本人另一个项目[“互联网新闻情感分析”](https://github.com/DefuLi/Emotional-Analysis-of-Internet-News)<br>

## 2 项目结构
项目文件夹共包括以下文件及文件夹：<br>
main.py 主程序，里面包括了自定义的全连接层，以及自定义了生成Dataset的子类。<br>
data 文件夹中包括了训练集和测试集，具体格式为id, text, label.<br>

## 3 网络结构
使用PyTroch自定义网络结构比较简单方便，由于是对句向量进行三分类，所以在句向量后自定义了一个全连接层。<br>
```python
网络类，全连接层
class Net(nn.Module):
    # in_dim=768, out_dim=3
    def __init__(self, in_dim, out_dim):
        super(Net, self).__init__()

        # 先放一层全连接层，后期可以多放几层
        self.fc = nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        out = self.fc(x)
        return out
```

## 4 自定义读取数据类
使用PyTorch可以继承Dataset类，重写__getitem__, __len__方法即可实现读取数据集的功能，生成的Dataset可以直接输入到DataLoader中，便可以对batch， shuffle进行自定义设置。<br>
```python
pytorch的dataset类 重写getitem,len方法
class Custom_dataset(Dataset):
    def __init__(self, dataset_list):
        self.dataset = dataset_list

    def __getitem__(self, item):
        text = self.dataset[item][1]
        label = self.dataset[item][2]

        return text, label

    def __len__(self):
        return len(self.dataset)
```

## 5 训练及测试
在训练集上训练完，将结果提交到比赛官网进行测试，结果显示F1值保持在0.7以上，比之前使用词表以及LSTM分类结果要高一些。<br>

## 6 注意事项
本程序使用Bert-As-Service进行句向量的生成，在运行本程序之前，一定要提前安装[Bert-As-Service库](https://github.com/hanxiao/bert-as-service)并在服务器启动。<br>
本程序使用了GPU加速以及visdom可视化，如果不需要，请在main.py文件中删除相关源代码。<br>

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
#导入库


#超参数
batch_size = 64  
#批处理大小
learning_rate = 0.01
#学习率是在BP算法中，描述参数更新幅度的一个参数
#学习率越大，越能够跳出局部最优解，但学习率过大会导致收敛速度变慢
#学习率过小，收敛速度变快，但可能跳不出局部最优解
momentum = 0.5
#动量参数，用于加速梯度下降过程，用以表达上一次参数更新对本次参数更新的影响程度
#能够帮助跳出局部最优解
EPOCH = 10


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
#Compose()函数可以用于组合多个转换函数，形成一个完整的转换流程。Compose()函数接受一个包含转换函数的列表作为参数，并返回一个函数对象
#ToTensor()函数将图片转化为张量，Normalize()函数对张量进行归一化处理。
train_dataset = datasets.MNIST(root='./data/mnist', train=True, download=False, transform=transform)  
test_dataset = datasets.MNIST(root='./data/mnist', train=False, download=False, transform=transform)  
# train=True训练集，=False测试集
#MNIST()函数用于加载MNIST数据集，root参数指定数据集的路径
#transform参数指定数据集的预处理方式（前已定义）
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)  #训练集数据
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)  #测试集数据
#DataLoader()函数用于加载数据集，batch_size指定每次加载的数据量，shuffle参数指定是否打乱数据
#使用DataLoader()函数的目的是将数据打乱，并对数据进行小批量的分批次处理，提高训练效率


#显示数据集中数据
fig = plt.figure()
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.tight_layout()
    #用以调整子图的位置，使其既不会重叠，也不会出现空白
    plt.imshow(train_dataset.data[i+100], cmap='binary', interpolation='none')
    #imshow()函数用于显示图片，接收一个数组参数，可以是灰度等，cmap参数指定颜色映射表，interpolation参数指定插值方式
    plt.title("Labels: {}".format(train_dataset.targets[i+100]))
    plt.axis('off')
    #将坐标轴的刻度清除
plt.show()


#构建神经网络
class Net(torch.nn.Module):
    #继承nn.Module类
    def __init__(self):
       super(Net,self).__init__()
       #调用父类构造函数
       self.conv1 = torch.nn.Sequential(
           torch.nn.Conv2d(1, 10, kernel_size=5),#卷积层
           torch.nn.ReLU(),#激励层
           torch.nn.MaxPool2d(kernel_size=2)#池化层
       )
       self.conv2=torch.nn.Sequential(
           torch.nn.Conv2d(10,20,kernel_size=5),
           torch.nn.ReLU(),
           torch.nn.MaxPool2d(kernel_size=2),
       )
       self.fc=torch.nn.Sequential(
           torch.nn.Linear(320,50),
           torch.nn.Linear(50,10),
           #两层全连接层，用于融合卷积层提取出的特征，并根据特征进行分类
       )
       #注意，conv1，conv2，fc都是类的属性
    def forward(self, x):#前向传递函数
        batch_size = x.size(0)#x是一个储存了图像数据的张量，size(0)返回张量的第一个维度，即图像的数量
        x = self.conv1(x)  
        x = self.conv2(x)#两次卷积-激励-池化
        x = x.view(batch_size, -1) #将张量x展平，即将张量的维度由（batch_size,10,28,28）变为（batch_size,320）
        x = self.fc(x)#两层全连接层
        return x  # 最后输出的是维度为10的，也就是（对应数学符号的0~9）
       
net = Net()
#net.load_state_dict(torch.load('num_model.pth'))

#损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()  
#交叉熵损失函数，描述预测概率分布与真实概率分布的差异
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)  
#随机梯度下降
#损失函数评价网络预测值与真实值之间的差异的大小
#优化函数是用来优化网络中的各个参数的，包括感知机的权重、阈值等，其目标就是使损失函数最小化


#训练模型
def train(epoch,train_data):
    running_loss = 0.0
    running_total = 0
    running_correct = 0
    for batch_idx, data in enumerate(train_data, 0):
        #使用enumerate()函数，既可以得到range序列，又可以得到与之对应的索引
        inputs, target = data
        optimizer.zero_grad()
        # zero_grad()函数用于将网络中的参数的梯度清零，防止梯度累加

        #前向反馈-->反向反馈-->参数更新
        outputs = net(inputs)#前向反馈
        loss = criterion(outputs, target)#计算误差
        loss.backward()#反向反馈
        optimizer.step()#参数更新

        running_loss += loss.item()
        _,predicted = torch.max(outputs.data, dim=1)
        # torch.max()函数用于返回张量中最大值，在这里就是返回张量中最大概率对应的数字，概率越大，则越可能是这个数字
        #output的第0维是图像数
        running_total += inputs.shape[0]
        running_correct += (predicted == target).sum().item()


        test_point=200
        if batch_idx % test_point == test_point-1:  #计算一批数据的平均损失和准确率
            print('[%d, %5d]: loss: %.3f , acc: %.2f %%'
                  % (epoch + 1, batch_idx + 1, running_loss / test_point, 100 * running_correct / running_total))
            running_loss = 0.0
            running_total = 0
            running_correct = 0


#测试模型
def test(test_data):
    correct = 0
    total = 0
    with torch.no_grad():#因为是测试集所以关闭梯度运算提高运算效率
        for data in test_data:
            images, labels = data
            outputs = net(images)
            _,predicted = torch.max(outputs.data, dim=1)
            total += labels.shape[0]
            correct += (predicted == labels).sum().item()
    acc = correct / total
    print(' Accuracy on test set: %.2f %% ' % (100 * acc))  # 求测试的准确率，正确数/总数
    return acc


acclist=[]
for epoch in range(EPOCH):  # 训练10个周期
    train(epoch, train_loader)
    acc = test(test_loader)
    acclist.append(acc)
    print('Epoch: %d, Accuracy: %.2f %%' %(epoch+1, acc*100))
torch.save(net.state_dict(), 'num_model.pth')
fig=plt.figure()
plt.plot(range(EPOCH),acclist)
plt.show()
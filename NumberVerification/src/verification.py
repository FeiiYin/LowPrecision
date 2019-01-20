# -*- encoding: utf-8
from VerificationCode.data_analysis import load_data
from VerificationCode.model import CNN
from torch import nn, optim, from_numpy, max, save
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset

EPOCH_NUM = 1  # 迭代次数
LEARNING_RATE = 0.02  # 学习率
BATCH_SIZE = 64  # 每批数量

# 加载数据
data_set = load_data()
train_dataset = TensorDataset(
    from_numpy(data_set[0]), from_numpy(data_set[1]))
train_loader = DataLoader(
    dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 初始化模型
cnn = CNN()
# print(cnn)

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 定义优化器
optimizer = optim.SGD(cnn.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCH_NUM):
    for i, (images, labels) in enumerate(train_loader):
        images = images.unsqueeze(1)
        images = Variable(images)
        labels = Variable(labels)
        out = cnn(images)
        loss = criterion(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 10 == 9:
            print("Batch %d is over" % i)
save(cnn, 'F:/cnn.pkl')

# -*- encoding: utf-8
from torch import nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=3),
            nn.BatchNorm2d(num_features=25),  # 小批量归一化
            nn.ReLU(inplace=True)  # 对输入运用修正线性单元函数，覆盖运算
        )
        self.layer_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)  # 池化窗口大小为2，步长为2
        )

        self.layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=25, out_channels=50, kernel_size=3),
            nn.BatchNorm2d(num_features=50),  # 小批量归一化
            nn.ReLU(inplace=True)  # 对输入运用修正线性单元函数，覆盖运算
        )

        self.layer_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=50 * 5 * 5, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=10)
        )

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


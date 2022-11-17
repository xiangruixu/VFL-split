import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable


class Client_AlexNet(nn.Module):  # 训练 ALexNet
    def __init__(self, num_classes=2):
        super(Client_AlexNet, self).__init__()
        # 五个卷积层 输入 32 * 32 * 3
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),  # (32-3+2)/1+1 = 32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # (32-2)/2+1 = 16
        )
        self.conv2 = nn.Sequential(  # 输入 16 * 16 * 6
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),  # (16-3+2)/1+1 = 16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # (16-2)/2+1 = 8
        )
        self.conv3 = nn.Sequential(  # 输入 8 * 8 * 16
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1),  # (8-3+2)/1+1 = 8
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # (8-2)/2+1 = 4
        )
        self.conv4 = nn.Sequential(  # 输入 4 * 4 * 64
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),  # (4-3+2)/1+1 = 4
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # (4-2)/2+1 = 2
        )
        self.conv5 = nn.Sequential(  # 输入 2 * 2 * 128
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),  # (2-3+2)/1+1 = 2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # (2-2)/2+1 = 1
        )  # 最后一层卷积层，输出 1 * 1 * 128
        # 全连接层
        self.dense = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size()[0], -1)
        x = self.dense(x)
        return x


class Server_AlexNet(nn.Module):
    def __init__(self,  channel=3, hideen1=64, hideen2=128, hideen3=256, hideen4=128, hideen5=64, num_classes=2):
        super(Server_AlexNet, self).__init__()
        act = nn.LeakyReLU
        self.fc2 = nn.Sequential(
            nn.Linear(hideen2, hideen3),
            nn.LeakyReLU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(hideen3, hideen4),
            nn.LeakyReLU(),
        )
        self.fc4 = nn.Sequential(
            nn.Linear(hideen4, hideen5),
        )
        self.fc5 = nn.Sequential(
            nn.Linear(hideen5, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2):
        x= torch.cat([x1, x2], dim=1)
        out = self.fc2(x)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.fc5(out)
        return out
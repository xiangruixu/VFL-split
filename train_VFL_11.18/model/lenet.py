# -*-coding:utf-8-*-
from torch import nn
import numpy as np
import torch


class LeNet(nn.Module):
    def __init__(self,  name=None, created_time=None, channel=3, hideen1=768, hideen2=1000, hideen3=768, num_classes=2):
        super(LeNet, self).__init__()
        act = nn.LeakyReLU
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(hideen1, hideen2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hideen2, hideen3)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(hideen3, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out


class Client_LeNet(nn.Module):
    def __init__(self,  name=None, created_time=None, channel=3, hideen1=768, hideen2= 128, hideen3=128, num_classes=2):
        super(Client_LeNet, self).__init__()
        act = nn.LeakyReLU
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(hideen1, hideen2)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out



class Server_LeNet(nn.Module):
    def __init__(self,  name=None, created_time=None, channel=3, hideen1=64, hideen2=128, hideen3=256, hideen4=128, hideen5=64, num_classes=2):
        super(Server_LeNet, self).__init__()
        act = nn.LeakyReLU
        self.fc2 = nn.Sequential(
            nn.Linear(hideen2, hideen3)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(hideen3, hideen4)
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





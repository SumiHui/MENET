# -*- coding: utf-8 -*-
# @File    : derain_pytorch/networks.py
# @Info    : @ TSMC-SIGGRAPH, 2019/4/16
# @Desc    : 
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -. 

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_ssim import SSIM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = [0]


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class DerainNet(nn.Module):
    """input shape: (n,c,h,w)"""

    def __init__(self):
        """
        Constructor of the architecture.

        """
        super(DerainNet, self).__init__()

        self.conv_bn1 = nn.Sequential(nn.Conv2d(3, 16, 3, padding=(1, 1), bias=False), nn.BatchNorm2d(16))
        self.act1 = nn.ReLU(True)
        self.conv_bn_act2 = nn.Sequential(nn.Conv2d(16, 16, 3, padding=(1, 1), bias=False),
                                          nn.BatchNorm2d(16), nn.ReLU(True))
        self.res_layer1 = ResidualBlock(16, 16)
        self.res_layer2 = ResidualBlock(16, 16)
        self.res_layer3 = ResidualBlock(16, 16)
        self.res_layer4 = ResidualBlock(16, 16)
        self.conv_bn3 = nn.Sequential(nn.Conv2d(16, 16, 3, padding=(1, 1), bias=False),
                                      nn.BatchNorm2d(16))
        self.act3 = nn.ReLU(True)
        self.conv_bn4 = nn.Sequential(nn.Conv2d(16, 3, 3, padding=(1, 1), bias=False),
                                      nn.BatchNorm2d(3))
        self.act4 = nn.Sigmoid()

    def forward(self, x):
        layer1_bn = self.conv_bn1(x)
        layer1 = self.act1(layer1_bn)
        layer2 = self.conv_bn_act2(layer1)

        res = self.res_layer1(layer2)
        res = self.res_layer2(res)
        res = self.res_layer3(res)
        res = self.res_layer4(res)

        layer3_bn = self.conv_bn3(res)
        layer3 = self.act3(layer1_bn + layer3_bn)
        layer4_bn = self.conv_bn4(layer3)
        layer4 = self.act4(x + layer4_bn)
        return layer4

    def get_last_shared_layer(self):
        return self.conv_bn4


class RegressionTrain(nn.Module):
    def __init__(self, model: DerainNet, n_tasks: int):
        """container for performing the training process
        :param model:
        :param n_tasks: number of tasks to solve
        """
        # initialize the module using super() constructor
        super(RegressionTrain, self).__init__()
        # print model architecture
        print(model)
        if os.path.exists("ckpt/checkpoint.data"):
            checkpoint = torch.load("ckpt/checkpoint.data")
            # print(checkpoint.keys())
            # print(checkpoint['state_dict'].keys())
            model.load_state_dict(checkpoint['state_dict'])  # restore model params
            print("==> restored checkpoint at '{}' epochs".format(checkpoint['epoch']))
        # model = nn.DataParallel(model, device_ids=device_ids).to(device)
        # assign the architectures
        self.model = model  # note: self.model=model.cuda(), if report "torch.cuda().Floattensor ..."

        # assign the weights for each task
        self.weights = nn.Parameter(torch.ones(n_tasks).float())
        # loss function
        self.mse_loss = nn.MSELoss().to(device)
        self.ssim = SSIM()
        # self.l1_loss = nn.L1Loss().to(device)

    def forward(self, x, y):
        y_hat = self.model(x)

        task_loss = []
        task_loss.append(self.mse_loss(y_hat, y))
        # task_loss.append(self.l1_loss(y_hat, y))
        task_loss.append(1 - self.ssim(y_hat, y))
        losses = torch.stack(task_loss)

        return losses

    def get_last_shared_layer(self):
        return self.model.get_last_shared_layer()


class SuperResolutionNet(nn.Module):
    """input shape: (n,c,h,w)"""

    def __init__(self, block=ResidualBlock, mid_channels=64, layers=4):
        super(SuperResolutionNet, self).__init__()
        self.in_channels = mid_channels
        self.conv_bn_act1 = nn.Sequential(nn.Conv2d(3, mid_channels, 9, padding=(4, 4), bias=False),
                                          nn.BatchNorm2d(mid_channels), nn.ReLU(True))
        self.res_layer = self.make_res_layer(block, mid_channels, layers)
        self.zoomin_layers = self.make_zoomin_layer()
        self.conv_bn_act2 = nn.Sequential(nn.Conv2d(mid_channels, 3, 9, padding=(4, 4), bias=False),
                                          nn.BatchNorm2d(3), nn.ReLU(True))

    def make_res_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels, kernel_size=3,
                                                 stride=stride, padding=1, bias=False),
                                       nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))  # 残差直接映射部分
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def make_zoomin_layer(self, in_channels=64, out_channels=64, num_layers=2, stride=2):
        layers = []
        for i in range(num_layers):
            layers.append(nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, 3, stride, 1, 1, bias=False),
                                        nn.BatchNorm2d(out_channels), nn.ReLU(True)))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.max_pool2d(x, (3, 3), 2, padding=1)
        x = F.max_pool2d(x, (3, 3), 2, padding=1)
        # print(x.size())
        y = self.conv_bn_act1(x)
        y = self.res_layer(y)
        y = self.zoomin_layers(y)
        y = self.conv_bn_act2(y)
        # print(y.size())
        return y

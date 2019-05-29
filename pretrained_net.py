# -*- coding: utf-8 -*-
# @File    : derain_pytorch/pretrained_net.py
# @Info    : @ TSMC-SIGGRAPH, 2019/4/27
# @Desc    : 
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -. 

"""VGG16
0 Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
1 ReLU(inplace)
2 Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
3 ReLU(inplace)
4 MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
5 Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
6 ReLU(inplace)
7 Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
8 ReLU(inplace)
9 MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
10 Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
11 ReLU(inplace)
12 Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
13 ReLU(inplace)
...
"""

# import torch
# import torch.nn as nn
# import torchvision
#
#
# class VGGNet(nn.Module):
#     def __init__(self):
#         """Select conv2_2 activation maps.  Download and load the pretrained vgg16."""
#         super(VGGNet, self).__init__()
#         self.select = ['8']
#         self.vgg = torchvision.models.vgg16(pretrained=True).features[:9]   # loc layer_8
#
#     def forward(self, x):
#         """Extract multiple convolutional feature maps."""
#         features = []
#         for name, layer in self.vgg._modules.items():
#             # print(name,layer)
#             x = layer(x)
#             if name in self.select:
#                 features.append(x)
#         return features
#
# # 设备配置
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# vgg = VGGNet().to(device).eval() # 切换到eval()模式，省去梯度计算量
#
# images = torch.randn(1, 3, 64, 64)
# # 提取多层特征向量
# target_features = vgg(images)
# print(len(target_features),type(target_features[0]))
# print(target_features[0].detach().numpy().shape)

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


img_to_tensor = transforms.ToTensor()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_vgg_model():
    model = models.vgg16(pretrained=True).features[:9]
    model = model.eval()
    model.to(device)
    return model


def extract_feature(model, img_tensor):
    model.eval()
    return model(img_tensor)


if __name__ == "__main__":
    model = make_vgg_model()
    imgpath = 'demo_label.jpg'

    TARGET_IMG_SIZE = 224
    img = Image.open(imgpath)  # 读取图片
    img = img.resize((TARGET_IMG_SIZE, TARGET_IMG_SIZE))
    tensor = img_to_tensor(img)  # 将图片转化成tensor
    tensor = tensor.view(1, 3, 224, 224)
    tensor = tensor.to(device)

    result_npy = extract_feature(model, tensor).data.cpu().numpy()  # 保存的时候一定要记得转成cpu形式的，不然可能会出错
    print(result_npy.shape)

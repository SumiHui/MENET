# -*- coding: utf-8 -*-
# @File    : derain_pytorch/random_net.py
# @Info    : @ TSMC-SIGGRAPH, 2019/4/29
# @Desc    : 
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -. 


import torch
from torch import nn


class RandomInitNet(nn.Module):
    def __init__(self):
        super(RandomInitNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, stride=2, padding=(2, 2))
        self.relu = nn.ReLU(True)
        torch.manual_seed(1)
        # nn.init.normal_(self.conv1.weight, mean=0, std=1)
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.normal_(self.conv1.bias)

    def forward(self, x):
        with torch.no_grad():
            y = self.relu(self.conv1(x))
            return y


model = RandomInitNet()

if __name__ == '__main__':
    from PIL import Image
    import torchvision.transforms as transforms
    import os
    from scipy.misc import imsave

    img_to_tensor = transforms.ToTensor()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imgpath = 'demo_label.jpg'

    TARGET_IMG_SIZE = 224
    img = Image.open(imgpath)  # 读取图片
    img = img.resize((TARGET_IMG_SIZE, TARGET_IMG_SIZE))
    tensor = img_to_tensor(img)  # 将图片转化成tensor
    tensor = tensor.view(1, 3, 224, 224)
    tensor = tensor.to(device)

    result_npy = model(tensor).data.cpu().numpy()
    print(result_npy.shape)

    savedir = "results"
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    for i in range(64):
        imsave("{}/{}.jpg".format(savedir,i), result_npy[0, i, ...])

# -*- coding: utf-8 -*-
# @File    : derain_pytorch/run_inference.py
# @Info    : @ TSMC-SIGGRAPH, 2019/4/16
# @Desc    : 
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -. 


import os

import matplotlib as mpl
mpl.use('Agg')
import torch
import torch.nn as nn

from data_helper_h5 import get_batch
from networks import DerainNet


from matplotlib import pyplot as plt
from configuration import cfg

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = [0]

torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    model = DerainNet()
    print(model)
    checkpoint = torch.load("ckpt/checkpoint.data")
    print(checkpoint.keys())
    print(checkpoint['state_dict'].keys())
    model.load_state_dict(checkpoint['state_dict'])  # restore model params
    print("==> loaded checkpoint '{}' epochs".format(checkpoint['epoch']))

    model = nn.DataParallel(model, device_ids=device_ids).to(device)

    criterion = nn.MSELoss()
    criterion.to(device)

    with torch.no_grad():
        torch.cuda.empty_cache()
        dataloader = get_batch()
        for i, sample in enumerate(dataloader):
            # print("input tensor shape: ", sample['input'].size())
            model.eval()

            # bs, ncrops, c, h, w = sample['input'].size()
            # image = sample['input'].view(-1, c, h, w)
            # image = image.to(device)
            x, y = sample
            x = torch.Tensor(x).to(device)
            y = torch.Tensor(y).to(device)

            pred = model(x)
            loss = criterion(pred, y)

            print("iter {}: loss {}".format(i, loss.data))

            input_crops = pred.numpy()
            label_crops = y.numpy()

            # p1 = plt.subplot(131)
            # p2 = plt.subplot(132)
            # p3 = plt.subplot(133)
            # p1.imshow(input_crops[0, ...].transpose((1, 2, 0)))
            # p2.imshow(label_crops[0, ...].transpose((1, 2, 0)))
            # p3.imshow(image.numpy()[0, ...].transpose((1, 2, 0)))
            # plt.show()
            save_dir="results"
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            for i in range(cfg.batch_size):
                plt.imsave("{}/{}_pred.jpg".format(save_dir,i), input_crops[i, ...].transpose((1, 2, 0)))
                plt.imsave("{}/{}_label.jpg".format(save_dir,i), label_crops[i, ...].transpose((1, 2, 0)))

            break

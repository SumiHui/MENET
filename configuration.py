# -*- coding: utf-8 -*-
# @File    : derain_pytorch/configuration.py
# @Info    : @ TSMC-SIGGRAPH, 2019/4/16
# @Desc    : 
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -. 


class ModelConfiguration(object):
    def __init__(self):
        self.lr = 1e-1  # initial learning rate
        self.batch_size = 10
        self.epochs = 10

        self.momentum = 0.9
        self.weight_decay = 0.99

        self.crop_size = 64

        # for balance loss
        self.mode = "grad_norm"
        self.alpha = 0.2


cfg = ModelConfiguration()

# -*- coding: utf-8 -*-
# @File    : derain_feqe_tf/net_shallow_vgg_fixed.py
# @Info    : @ TSMC-SIGGRAPH, 2019/11/15
# @Desc    :
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -.


import tensorflow as tf

from configuration import cfg
from template import menet_shallow_new
from vgg19 import Vgg19


class ModelShallowNewVGGFixed(menet_shallow_new.ModelShallowNew):
    def __init__(self, mode):
        """
        :param mode: one of strings "train", "eval", "inference"
        """
        super(ModelShallowNewVGGFixed, self).__init__(mode)

    # loss_layer
    def build_loss(self):
        # Compute losses.
        self.mse = tf.losses.mean_squared_error(labels=self.bg_img, predictions=self.output)

        perceptron = Vgg19(cfg.vgg_dir)
        perceptron.build(tf.concat([self.bg_img, self.output], axis=0))
        self.content_loss = tf.losses.mean_squared_error(perceptron.conv3_4[:cfg.batch_size],
                                                         perceptron.conv3_4[cfg.batch_size:])

        self.ssim = tf.reduce_mean(tf.image.ssim(self.bg_img, self.output, max_val=255.0))
        self.psnr = tf.reduce_mean(tf.image.psnr(self.bg_img, self.output, max_val=255.0))
        self.total_loss = self.mse + 1e-3 * self.content_loss

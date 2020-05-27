# -*- coding: utf-8 -*-
# @File    : derain_feqe_tf/net_shallow_edge_fixed.py
# @Info    : @ TSMC-SIGGRAPH, 2019/11/15
# @Desc    :
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -.


import tensorflow as tf

from configuration import cfg
from template import menet_shallow_new


class ModelShallowNewEdgeFixed(menet_shallow_new.ModelShallowNew):
    def __init__(self, mode):
        """
        :param mode: one of strings "train", "eval", "inference"
        """
        super(ModelShallowNewEdgeFixed, self).__init__(mode)

    # loss_layer
    def build_loss(self):
        # Compute losses.
        self.mse = tf.losses.mean_squared_error(labels=self.bg_img, predictions=self.output)

        edge_feat = tf.image.sobel_edges(tf.concat([self.bg_img, self.output], axis=0))
        self.content_loss = tf.losses.mean_squared_error(labels=edge_feat[:cfg.batch_size],
                                                         predictions=edge_feat[cfg.batch_size:])

        self.ssim = tf.reduce_mean(tf.image.ssim(self.bg_img, self.output, max_val=255.0))
        self.psnr = tf.reduce_mean(tf.image.psnr(self.bg_img, self.output, max_val=255.0))
        self.total_loss = self.mse + 1e-2 * self.content_loss

# -*- coding: utf-8 -*-
# @File    : derain_feqe_tf/net_shallow_edge_gradbalance.py
# @Info    : @ TSMC-SIGGRAPH, 2019/11/13
# @Desc    :
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -.


import tensorflow as tf

from configuration import cfg
from template import menet_shallow_new


class ModelShallowNewEdgeGradBalance(menet_shallow_new.ModelShallowNew):
    def __init__(self, mode):
        """
        :param mode: one of strings "train", "eval", "inference"
        """
        super(ModelShallowNewEdgeGradBalance, self).__init__(mode)

    # loss_layer
    def build_loss(self):
        # Compute losses.
        self.mse = tf.losses.mean_squared_error(labels=self.bg_img, predictions=self.output)

        edge_feat = tf.image.sobel_edges(tf.concat([self.bg_img, self.output], axis=0))
        self.content_loss = tf.losses.mean_squared_error(labels=edge_feat[:cfg.batch_size],
                                                         predictions=edge_feat[cfg.batch_size:])

        self.ssim = tf.reduce_mean(tf.image.ssim(self.bg_img, self.output, max_val=255.0))
        self.psnr = tf.reduce_mean(tf.image.psnr(self.bg_img, self.output, max_val=255.0))
        self.total_loss = self.mse + self.content_loss

    def build_optimizer(self):
        # the loss ratio for task i at time t
        tvars = tf.trainable_variables(scope="derain/layer5")
        mse_grads = tf.gradients(self.mse, tvars)
        G1 = tf.norm(mse_grads)

        closs_grads = tf.gradients(self.content_loss, tvars)
        G2 = tf.norm(closs_grads)

        G = G1 + G2
        w_1 = tf.stop_gradient(1. - G1 / G)  # num_tasks * (1 - task_i/tasks)
        w_2 = tf.stop_gradient(1. - G2 / G)

        self.total_loss = w_1 * self.mse + w_2 * self.content_loss

        # note: cfg.num_examples_per_epoch now is `None`
        lr = tf.train.exponential_decay(cfg.lr,
                                        self.global_step,
                                        cfg.num_examples_per_epoch // cfg.batch_size * cfg.decay_epochs,
                                        cfg.decay_factor,
                                        staircase=True)
        optimizer = tf.train.AdamOptimizer(lr)
        self.lr = optimizer._lr

        # note: you must use the control dependency to update the BN parameters.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.total_loss, self.global_step)

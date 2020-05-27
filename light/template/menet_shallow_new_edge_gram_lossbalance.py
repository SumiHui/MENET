# -*- coding: utf-8 -*-
# @File    : derain_feqe_tf/net_deep_color_lossbalance.py
# @Info    : @ TSMC-SIGGRAPH, 2019/11/26
# @Desc    :
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -.


import tensorflow as tf

from configuration import cfg
from template import menet_shallow_new
# from vgg19 import Vgg19

class ModelShallowNewEdgeGramLossBalance(menet_shallow_new.ModelShallowNew):
    def __init__(self, mode):
        """
        :param mode: one of strings "train", "eval", "inference"
        """
        super(ModelShallowNewEdgeGramLossBalance, self).__init__(mode)
        self.edge_loss = None

    # def texture_matching_loss(self):
    #     perceptron = Vgg19(cfg.vgg_dir)
    #     perceptron.build(tf.concat([self.bg_img, self.output], axis=0))
    #     labels_reshape = tf.reshape(perceptron.pool1[:cfg.batch_size], [cfg.batch_size, -1, 64])
    #     predictions_reshape = tf.reshape(perceptron.pool1[cfg.batch_size:], [cfg.batch_size, -1, 64])
    #     gram_labels = tf.matmul(tf.transpose(labels_reshape, [0, 2, 1]), labels_reshape)
    #     gram_predictions = tf.matmul(tf.transpose(predictions_reshape, [0, 2, 1]), predictions_reshape)
    #     gram_labels = tf.reduce_mean(gram_labels, [1,2])
    #     gram_predictions = tf.reduce_mean(gram_predictions, [1,2])
    #     # texture_matching_loss
    #     return tf.losses.mean_squared_error(labels=gram_labels, predictions=gram_predictions)

    def texture_matching_loss(self, labels, predictions):
        labels_reshape = tf.reshape(tf.space_to_depth(labels, 4), [cfg.batch_size, -1, 48])
        predictions_reshape = tf.reshape(tf.space_to_depth(predictions, 4), [cfg.batch_size, -1, 48])
        gram_labels = tf.matmul(tf.transpose(labels_reshape, [0, 2, 1]), labels_reshape)
        gram_predictions = tf.matmul(tf.transpose(predictions_reshape, [0, 2, 1]), predictions_reshape)
        gram_labels = tf.reduce_mean(gram_labels, [1,2])
        gram_predictions = tf.reduce_mean(gram_predictions, [1,2])
        # texture_matching_loss
        return tf.losses.mean_squared_error(labels=gram_labels, predictions=gram_predictions)

    # loss_layer
    def build_loss(self):
        # Compute losses.
        self.mse = tf.losses.mean_squared_error(labels=self.bg_img, predictions=self.output)

        edge_feat = tf.image.sobel_edges(tf.concat([self.bg_img, self.output], axis=0))
        self.edge_loss = tf.losses.mean_squared_error(labels=edge_feat[:cfg.batch_size],
                                                        predictions=edge_feat[cfg.batch_size:])

        # self.content_loss = self.texture_matching_loss()
        self.content_loss = self.texture_matching_loss(labels=self.bg_img, predictions=self.output)

        self.ssim = tf.reduce_mean(tf.image.ssim(self.bg_img, self.output, max_val=255.0))
        self.psnr = tf.reduce_mean(tf.image.psnr(self.bg_img, self.output, max_val=255.0))
        self.total_loss = self.mse + self.content_loss + self.edge_loss

    def build_optimizer(self):
        # the loss ratio for task i at time t
        w_1 = tf.stop_gradient(1. - self.mse / self.total_loss)
        w_2 = tf.stop_gradient(1. - self.content_loss / self.total_loss)
        w_3 = tf.stop_gradient(1. - self.edge_loss / self.total_loss)

        self.total_loss = w_1 * self.mse + w_2 * self.content_loss + w_3 * self.edge_loss

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

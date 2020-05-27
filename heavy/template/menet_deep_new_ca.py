# -*- coding: utf-8 -*-
# @File    : light/net_shallow_new_ca.py
# @Info    : @ TSMC-SIGGRAPH, 2019/11/13
# @Desc    : deep model (16 residual blocks), spatial pyramid attention
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -.


import tensorflow as tf

from template import net_base


class ModelDeepNewCa(net_base.ModelBase):
    def __init__(self, mode):
        """
        :param mode: one of strings "train", "eval", "inference"
        """
        super(ModelDeepNewCa, self).__init__(mode)

    def channel_attention_layer(self, name, x):
        with tf.variable_scope(name):
            in_channel = x.get_shape()[-1]
            down_scale1 = self.bn_relu(self.conv2(tf.space_to_depth(x, 2, name="att_desubpixel_1"), in_channel, 1, 1, name="squeeze_1"))
            down_scale1 = self.bn_relu(self.conv2(down_scale1, in_channel, 3, name="layer_1"))
            down_scale2 = self.bn_relu(self.conv2(tf.space_to_depth(down_scale1, 2, name="att_desubpixel_2"), in_channel, 1, 1, name="squeeze_2"))
            down_scale2 = self.bn_relu(self.conv2(down_scale2, in_channel, 3, name="layer_2"))

            net = self.bn(self.conv2(down_scale2, in_channel, 3, 1, name="excitation_1"))
            channel_feat = tf.nn.sigmoid(tf.reduce_mean(net, [1, 2], keepdims=True))

            return tf.add(x, tf.multiply(x, channel_feat))

    def build_model(self):
        with tf.variable_scope("derain"):
            net = tf.space_to_depth(self.syn_img, 2, name="desubpixel_1")
            net_1 = self.bn(self.conv2(net, 16, 3, name="layer1"))
            net = tf.nn.relu(net_1)
            net = tf.space_to_depth(net, 2, name="desubpixel_2")
            net_2 = self.bn(self.conv2(net, 64, 3, name="layer2"))
            net = tf.nn.relu(net_2)

            net = self.channel_attention_layer("ca", net)

            for i in range(16):
                res = net
                net = self.bn_relu(self.conv2(net, 64, 3, 1, name='res_{}_a'.format(i)))
                net = self.bn(self.conv2(net, 64, 3, 1, name='res_{}_b'.format(i)))
                if i <15:
                    net = tf.nn.relu(tf.add(net, res))  # skip-connect
                else:
                    net = tf.add(net, res)

            net = self.bn(self.conv2(tf.add(net_2, net), 64, 3, name="layer3"))
            net = tf.depth_to_space(net, 2, "pixel_shuffle_1")
            net = self.bn(self.conv2(tf.add(net_1, net), 16, 3, name="layer4"))
            net = self.conv2(net, 12, 3, name="layer5")
            net = tf.depth_to_space(net, 2, "pixel_shuffle_2")

            bg_hat = tf.add(self.syn_img, net)
            self.output = tf.clip_by_value(bg_hat, 0.0, 255.0, name="output")  # BReLU



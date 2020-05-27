# -*- coding: utf-8 -*-
# @File    : derain_feqe_tf/net_base.py
# @Info    : @ TSMC-SIGGRAPH, 2019/11/13
# @Desc    :
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -.

import tensorflow as tf

from configuration import cfg


class ModelBase(object):
    def __init__(self, mode):
        """
        :param mode: one of strings "train", "eval", "inference"
        """
        assert mode in ["train", "eval", "inference"]
        self.initializer = tf.initializers.variance_scaling(scale=1.0, mode="fan_in")
        self.mode = mode

        # A float32 tensor with shape [batch_size, height, width, channels].
        self.bg_img = None  # clean background image
        self.syn_img = None  # synthesis rainy image
        self.r_img = None  # rain layer

        # Outputs of de-rain model
        self.output = None
        self.r_hat = None
        self.syn_hat = None

        # A float32 scalar tensor; the total loss for the trainer to optimize.
        self.total_loss = None
        self.mse = None
        self.content_loss = None
        self.contexture_loss = None

        self.ssim = None
        self.psnr = None
        self.lr = None

        # optimizer
        self.train_op = None

        # Global step tensor.
        self.global_step = None

        # class name
        self.nickname = self.__class__.__name__

    def is_training(self):
        """returns true if the model is built for training mode."""
        return self.mode == "train"

    def build_inputs(self):
        """Input prefetching, preprocessing and batching.
        :return:
        self.images: A tensor of shape [batch_size, height, width, channels].
        """
        if self.mode == "inference":
            # # In inference mode, images are fed via placeholders.
            self.syn_img = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3], name="image_feed")

            # No target of input rainy image in inference mode.
            self.bg_img = None
            self.r_img = None
        else:
            # from h5py get batch-data
            self.bg_img = tf.placeholder(tf.float32, shape=(cfg.batch_size, cfg.crop_size, cfg.crop_size, 3),
                                         name='bg')
            # self.r_img = tf.placeholder(tf.float32, shape=(cfg.batch_size, cfg.crop_size, cfg.crop_size, 1), name='r')
            self.syn_img = tf.placeholder(tf.float32, shape=(cfg.batch_size, cfg.crop_size, cfg.crop_size, 3),
                                          name='syn')

    def conv2(self, inputs, filters, kernel_size, strides=1, dilation_rate=1, activation=None, padding="SAME", name=None):
        with tf.variable_scope(name):
            assert type(strides) == int
            assert type(kernel_size) == int
            return tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
                                    padding=padding, dilation_rate=dilation_rate, activation=activation,
                                    kernel_initializer=self.initializer, use_bias=False, name=name + "_conv")

    @staticmethod
    def instance_norm(inputs):
        ins_mean, ins_sigma = tf.nn.moments(inputs, axes=[1, 2], keep_dims=True)
        return (inputs - ins_mean) / (tf.sqrt(ins_sigma + 1e-5))

    def bn(self, inputs):
        return tf.layers.batch_normalization(inputs=inputs, training=self.is_training())

    def bn_relu(self, inputs):
        return tf.nn.relu(tf.layers.batch_normalization(inputs=inputs, training=self.is_training()))

    def bn_lrelu(self, inputs):
        return tf.nn.leaky_relu(tf.layers.batch_normalization(inputs=inputs, training=self.is_training()))

    def in_relu(self, inputs):
        return tf.nn.relu(self.instance_norm(inputs))

    def in_lrelu(self, inputs):
        return tf.nn.leaky_relu(self.instance_norm(inputs))

    def build_model(self):
        pass

    @staticmethod
    def tf_summary_image(name, img_tensor, img_size=cfg.crop_size):
        v = tf.reshape(img_tensor[:4, :, :, :], [2, 2, img_size, img_size, 3])
        v = tf.transpose(v, [0, 2, 1, 3, 4])
        v = tf.reshape(v, [-1, 2 * img_size, 2 * img_size, 3])
        tf.summary.image(name, v)

    # loss_layer
    def build_loss(self):
        # Compute losses.
        self.mse = tf.losses.mean_squared_error(labels=self.bg_img, predictions=self.output)
        self.ssim = tf.reduce_mean(tf.image.ssim(self.bg_img, self.output, max_val=255.0))
        self.psnr = tf.reduce_mean(tf.image.psnr(self.bg_img, self.output, max_val=255.0))
        self.total_loss = self.mse

    def setup_global_step(self):
        """Sets up the global step tensor."""
        global_step = tf.Variable(initial_value=0, trainable=False, name="global_step",
                                  collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
        self.global_step = global_step

    def build_optimizer(self):
        # note: cfg.num_examples_per_epoch now is `None`
        lr = tf.train.exponential_decay(cfg.lr,
                                        self.global_step,
                                        cfg.num_examples_per_epoch // cfg.batch_size * cfg.decay_epochs,
                                        cfg.decay_factor,
                                        staircase=True)
        optimizer = tf.train.AdamOptimizer(lr)
        self.lr = optimizer._lr
        #optimizer = tf.train.MomentumOptimizer(lr,0.9)
        #self.lr = optimizer._learning_rate

        # note: you must use the control dependency to update the BN parameters.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.total_loss, self.global_step)

    def build(self):
        """Creates all ops for training and evaluation."""
        self.build_inputs()
        self.build_model()

        self.setup_global_step()

        if self.mode != "inference":
            self.build_loss()
            self.build_optimizer()

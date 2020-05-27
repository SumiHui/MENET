# -*- coding: utf-8 -*-
# @File    : derain_wgan_tf/train.py
# @Info    : @ TSMC-SIGGRAPH, 2019/8/10
# @Desc    :
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -.


import os

import numpy as np
import tensorflow as tf

from configuration import cfg
from data_helper import get_batch
from net import Model

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id


def main(_):
    # build model
    model = Model("train")
    model.build()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(max_to_keep=cfg.max_checkpoints_to_keep)

        if os.path.exists(os.path.join(cfg.model_dir, model.nickname, "checkpoint")):
            model_file = tf.train.latest_checkpoint(os.path.join(cfg.model_dir, model.nickname))
            saver.restore(sess, model_file)
        else:
            if not os.path.exists(os.path.join(cfg.model_dir, model.nickname)):
                os.makedirs(os.path.join(cfg.model_dir, model.nickname))
        # training loop
        for epoch in range(cfg.epochs):
            # iterate the whole dataset n epochs
            print("iterate the whole dataset {} epochs".format(cfg.epochs))
            for i, samples in enumerate(get_batch(os.path.join(cfg.train_dir, cfg.data_filename), cfg.batch_size, True)):
                batch_syn, batch_bg = samples
                step = tf.train.global_step(sess, model.global_step)
                batch_syn = np.asarray(batch_syn, "float32")
                batch_bg = np.asarray(batch_bg, "float32")
                feed_dict = {model.bg_img: batch_bg, model.syn_img: batch_syn}

                if step % cfg.num_steps_per_display == 0:
                    _, lr, total_loss, mse, ssim, psnr = sess.run([model.train_op, model.lr, model.total_loss, model.mse,
                                                                   model.ssim, model.psnr],
                                                                   feed_dict=feed_dict)
                    print("[{}/{}] lr: {:.8f}, total_loss: {:.6f}, mse: {:.6f}, ssim: {:.4f}, "
                          "psnr: {:.4f}".format(epoch, step, lr, total_loss, mse, ssim, psnr))
                else:
                    sess.run(model.train_op, feed_dict=feed_dict)
            saver.save(sess, os.path.join(cfg.model_dir, model.nickname, 'model.epoch-{}'.format(epoch)))
        saver.save(sess, os.path.join(cfg.model_dir, model.nickname, 'model.final-{}'.format(cfg.epochs)))
        print(" ------ Arriving at the end of data ------ ")


if __name__ == '__main__':
    tf.app.run()

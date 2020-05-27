# -*- coding: utf-8 -*-
# @File    : derain_wgan_tf/validation.py
# @Info    : @ TSMC-SIGGRAPH, 2019/5/30
# @Desc    : 
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -. 


import os
import platform
from datetime import datetime
from time import time

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from configuration import cfg
from data_helper import get_batch
from net import Model

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id


def main(_):
    # build model
    model = Model("eval")
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
            exit()

        ssim_list = list()
        psnr_list = list()
        mse_list = list()
        time_list = list()
        for batch_syn, batch_bg in tqdm(get_batch(os.path.join(cfg.test_dir, cfg.data_filename), cfg.batch_size)):
            batch_syn = np.asarray(batch_syn, "float32")
            batch_bg = np.asarray(batch_bg, "float32")
            feed_dict = {model.bg_img: batch_bg, model.syn_img: batch_syn}

            start = time()
            mse, ssim, psnr = sess.run([model.mse, model.ssim, model.psnr], feed_dict=feed_dict)
            end = time()

            ssim_list.append(ssim)
            psnr_list.append(psnr)
            mse_list.append(mse)
            time_list.append(end - start)

        avg_ssim = np.mean(ssim_list)
        avg_psnr = np.mean(psnr_list)
        avg_mse = np.mean(mse_list)
        avg_time = np.mean(time_list) / cfg.batch_size

        if not os.path.exists(cfg.metric_dir):
            os.makedirs(cfg.metric_dir)

        with open(os.path.join(cfg.metric_dir, 'metrics.txt'), 'a') as f:
            f.write("os:\t{}\t\t\tdate:\t{}\n".format(platform.system(), datetime.now()))
            f.write("model:\t{}\t\timage_size:\t{}\n".format(model.nickname, cfg.crop_size))
            f.write("data:\t{}\t\tgpu_id:\t{}\n".format(cfg.data_filename, cfg.gpu_id))
            f.write("speed:\t{:.8f} s/item\tmse:\t{:.8f}\n".format(avg_time, avg_mse))
            f.write("ssim:\t{:.8f}\t\tpsnr:\t{:.8f}\n\n".format(avg_ssim, avg_psnr))

        print(" ------ Arriving at the end of data ------ ")


if __name__ == '__main__':
    tf.app.run()

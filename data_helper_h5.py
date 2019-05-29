# -*- coding: utf-8 -*-
# @File    : derain_gradnorm/data_helper_h5.py
# @Info    : @ TSMC-SIGGRAPH, 2019/4/27
# @Desc    :
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -.


import h5py
import numpy as np
from matplotlib import pyplot as plt
from random import shuffle
from configuration import cfg


def get_batch(filename="RainTrainL.h5", batch_size=cfg.batch_size, is_shuffle=True):
    f = h5py.File(filename, "r")
    inputs = f["input"]
    labels = f["label"]
    num_samples = inputs.len()
    num_batches = num_samples // batch_size
    if num_samples % batch_size != 0:
        num_samples = num_batches * batch_size
        inputs = inputs[:num_samples, ...]
        labels = labels[:num_samples, ...]

    if is_shuffle:
        idx = np.arange(num_samples)
        shuffle(idx)
        inputs = np.take(inputs, idx, 0)
        labels = np.take(labels, idx, 0)

    for i in range(num_batches):
        batch_x = inputs[i * batch_size:i * batch_size + batch_size, ...]
        batch_y = labels[i * batch_size:i * batch_size + batch_size, ...]
        yield batch_x, batch_y


if __name__ == '__main__':
    for batch_x, batch_y in get_batch("dataset/RainTrainL.h5", 16):
        img1 = np.transpose(batch_x, [0, 2, 3, 1])
        for i in range(16):
            a = plt.subplot(4, 8, i + 1)
            a.imshow(img1[i])

        img1 = np.transpose(batch_y, [0, 2, 3, 1])
        for i in range(16):
            a = plt.subplot(4, 8, i + 17)
            a.imshow(img1[i])
        plt.show()
        break

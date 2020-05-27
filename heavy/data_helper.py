# -*- coding: utf-8 -*-
# @File    : derain_gradnorm_tf/data_helper.py
# @Info    : @ TSMC-SIGGRAPH, 2019/5/29
# @Desc    :
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -.


from random import shuffle

import h5py
import numpy as np

from configuration import cfg


# from matplotlib import pyplot as plt


def get_batch(filename="dataset/derain_h5/RainTrainL.h5", batch_size=cfg.batch_size, is_shuffle=True):
    f = h5py.File(filename, "r")
    inputs = f["syn"]
    labels = f["bg"]
    num_samples = inputs.len()
    num_batches = num_samples // batch_size
    if num_samples % batch_size != 0:
        num_samples = num_batches * batch_size
        inputs = inputs[:num_samples, ...]
        labels = labels[:num_samples, ...]

    cfg.num_examples_per_epoch = num_samples
    print("[get_batch] processing {} samples, batch_size {}, batches {}".format(num_samples, batch_size, num_batches))

    idx = np.arange(num_samples)

    if is_shuffle:
        shuffle(idx)
        # Note: to avoid OOM, it is not recommended to shuffle the data in the following deprecated way
        # Deprecation method example: inp = np.take(inputs, idx, 0)

    for i in range(num_batches):
        # np.sort() for avoiding TypeError: Indexing elements must be in increasing order.
        batch_x = inputs[np.sort(idx[i * batch_size:i * batch_size + batch_size]), ...]
        batch_y = labels[np.sort(idx[i * batch_size:i * batch_size + batch_size]), ...]
        yield batch_x, batch_y


if __name__ == '__main__':
    for batch_x, batch_y in get_batch("dataset/derain_h5/Rain100L.h5", 4):
        # for i in range(4):
        #    a = plt.subplot(2, 4, i + 1)
        #    a.imshow(batch_x[i])
        #    a.axis('off')

        # for i in range(4):
        #    a = plt.subplot(2, 4, i + 5)
        #    a.imshow(batch_y[i])
        #    a.axis('off')
        # plt.show()
        print("test data helper", batch_x.shape, batch_y.shape)
        break

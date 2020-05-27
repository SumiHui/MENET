# -*- coding: utf-8 -*-
# @File    : derain_gradnorm_tf/build_h5_dataset.py
# @Info    : @ TSMC-SIGGRAPH, 2019/5/29
# @Desc    : 
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -. 


import os
from random import shuffle

import h5py
import numpy as np
from PIL import Image

from configuration import cfg
from utils import transforms


class DataLoader(object):
    """Construct a generator"""

    def __init__(self, image_dir, crop_size=64, blend_mode="linear", horizontal_flip=False):
        """
        :param image_dir (str): path of the images
        :param crop_size (int or tuple): crop size, default is 64
        :param blend_mode (str): pick on of two, `screen` or `linear`, represents image composition type
        :param horizontal_flip (bool): Whether use horizontal flipping or not
        """
        # super(DataLoader, self).__init__()
        # 1. initialize file path or a list of file names.
        assert blend_mode in ["screen", "linear"]
        self.blend_mode = blend_mode
        self.data_path = image_dir
        self.all_filenames = os.listdir(self.data_path)
        self.label_filenames = list(filter(lambda filename: filename.startswith("norain"), self.all_filenames))
        self.num_files = len(self.label_filenames)
        print("[DataLoader] preprocess {} files on dir `{}`".format(self.num_files, self.data_path))
        self.transform = transforms.Compose([transforms.FiveCrop(crop_size, horizontal_flip),  # tuple (tl, tr, bl, br, center)
                                             lambda crops: np.stack([transforms.ToArray()(crop) for crop in crops])])

    def __getitem__(self, item):
        # 1. read one data from file (e.g. using PIL.Image.open).
        # 2. Preprocess the data (e.g. Transform).
        # 3. Return a data pair (e.g. image and label).
        if self.blend_mode == "screen":
            input_image = Image.open(os.path.join(self.data_path,
                                                  self.label_filenames[item].replace("norain", "screenrainy")))
        else:
            input_image = Image.open(os.path.join(self.data_path,
                                                  self.label_filenames[item].replace("norain", "rain")))
        label_image = Image.open(os.path.join(self.data_path, self.label_filenames[item]))
        noise_image = Image.open(os.path.join(self.data_path, self.label_filenames[item].replace("norain", "rainstreak")))
        sample = {'syn': input_image, 'bg': label_image, 'r': noise_image}

        if self.transform:
            sample['syn'] = self.transform(sample['syn'])
            sample['bg'] = self.transform(sample['bg'])
            sample['r'] = self.transform(sample['r'])
        return sample

    def __len__(self):
        # the total size of dataset.(number of samples)
        return self.num_files


def save2h5(save_path="temp.h5", image_dir="/dataset/cvpr2017_derain_dataset/training_data/RainTrainL",
            crop_size=224, blend_mode="linear", horizontal_flip=False):
    dataloader = DataLoader(image_dir, crop_size, blend_mode, horizontal_flip)
    img_pair = []
    for samples in dataloader:
        samples['r'] = np.expand_dims(samples['r'], -1)
        img_pair.append(np.concatenate([samples['syn'], samples['bg'], samples['r']], -1))
    img_pair_ndarray = np.concatenate(img_pair, 0)
    idx = np.arange(img_pair_ndarray.shape[0])
    shuffle(idx)
    img_pair_ndarray = np.take(img_pair_ndarray, idx, 0)

    input_ndarray, label_ndarray, noise_ndarray = np.split(img_pair_ndarray, [3, 6], -1)

    dirname = os.path.dirname(save_path)
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    f = h5py.File(save_path, 'w')
    _ = f.create_dataset("syn", data=input_ndarray, compression="gzip")
    _ = f.create_dataset("bg", data=label_ndarray, compression="gzip")
    _ = f.create_dataset("r", data=noise_ndarray, compression="gzip")
    f.close()


if __name__ == '__main__':
    img_dir = os.path.join(cfg.original_image_dir, cfg.sub_dir)
    save2h5("{}.h5".format(os.path.join(cfg.test_dir, cfg.sub_dir)), img_dir, cfg.crop_size, "linear",
            cfg.horizontal_flip)

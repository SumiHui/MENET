# -*- coding: utf-8 -*-
# @File    : derain_pytorch/build_hdf5_dataset.py
# @Info    : @ TSMC-SIGGRAPH, 2019/4/27
# @Desc    : 
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -. 

import os
from random import shuffle

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from configuration import cfg


class CustomDataset(Dataset):
    def __init__(self, image_dir="/dataset/cvpr2017_derain_dataset/training_data/RainTrainL", blend_mode="linear"):
        # super(CustomDataset, self).__init__()
        # 1. initialize file path or a list of file names.
        assert blend_mode in ["screen", "linear"]
        self.blend_mode = blend_mode
        self.train_data_path = image_dir
        self.train_filenames = os.listdir(self.train_data_path)
        self.train_label_filenames = list(filter(lambda filename: filename.startswith("norain"), self.train_filenames))
        self.transform = transforms.Compose([transforms.CenterCrop(cfg.crop_size), transforms.ToTensor()])

    def __getitem__(self, item):
        # 1. read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        if self.blend_mode == "screen":
            input_image = Image.open(os.path.join(self.train_data_path,
                                                  self.train_label_filenames[item].replace("norain", "screenrainy")))
        else:
            input_image = Image.open(os.path.join(self.train_data_path,
                                                  self.train_label_filenames[item].replace("norain", "rain")))
        label_image = Image.open(os.path.join(self.train_data_path, self.train_label_filenames[item]))
        sample = {'input': input_image, 'label': label_image}

        if self.transform:
            sample['input'] = self.transform(sample['input'])
            sample['label'] = self.transform(sample['label'])
        return sample

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.train_label_filenames)


class CustomDatasetFive(Dataset):
    def __init__(self, image_dir="/dataset/cvpr2017_derain_dataset/training_data/RainTrainL", blend_mode="linear"):
        # super(CustomDataset, self).__init__()
        # 1. initialize file path or a list of file names.
        assert blend_mode in ["screen", "linear"]
        self.blend_mode = blend_mode
        self.train_data_path = image_dir
        self.train_filenames = os.listdir(self.train_data_path)
        self.train_label_filenames = list(filter(lambda filename: filename.startswith("norain"), self.train_filenames))
        self.num_files = len(self.train_label_filenames)
        print("[CustomDataset] preprocess {} files".format(self.num_files))
        self.transform = transforms.Compose([transforms.FiveCrop(cfg.crop_size),  # tuple (tl, tr, bl, br, center)
                                             lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])])

    def __getitem__(self, item):
        # 1. read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        if self.blend_mode == "screen":
            input_image = Image.open(os.path.join(self.train_data_path,
                                                  self.train_label_filenames[item].replace("norain", "screenrainy")))
        else:
            input_image = Image.open(os.path.join(self.train_data_path,
                                                  self.train_label_filenames[item].replace("norain", "rain")))
        label_image = Image.open(os.path.join(self.train_data_path, self.train_label_filenames[item]))
        sample = {'input': input_image, 'label': label_image}

        if self.transform:
            sample['input'] = self.transform(sample['input'])
            sample['label'] = self.transform(sample['label'])
        return sample

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.train_label_filenames)


def get_batch(image_dir="/dataset/cvpr2017_derain_dataset/training_data/RainTrainL", blend_mode="linear"):
    custom_dataset = CustomDatasetFive(image_dir, blend_mode)
    dataloader = DataLoader(dataset=custom_dataset, batch_size=cfg.batch_size, shuffle=True)
    return dataloader


if __name__ == '__main__':
    base_dir = "/dataset/cvpr2017_derain_dataset/training_data"
    datatype = "RainTrainL"
    if not os.path.exists("dataset"):
        os.mkdir("dataset")
    dataloader = get_batch(os.path.join(base_dir, datatype))
    img_pair = []
    for i, sample in enumerate(dataloader):
        bs, ncrops, c, h, w = sample['input'].size()
        sample['input'] = sample['input'].view(-1, c, h, w)
        sample['label'] = sample['label'].view(-1, c, h, w)
        pair_crops = torch.cat((sample['input'], sample['label']), 1)
        img_pair.append(pair_crops.numpy())

    img_ndarray = np.concatenate(img_pair, 0)
    idx = np.arange(img_ndarray.shape[0])
    shuffle(idx)
    img_ndarray = np.take(img_ndarray, idx, 0)

    input_ndarray, label_ndarray = np.split(img_ndarray, [3], 1)

    f = h5py.File("dataset/{}.h5".format(datatype), 'w')
    input_dataset = f.create_dataset("input", data=input_ndarray)
    label_dataset = f.create_dataset("label", data=label_ndarray)
    f.close()

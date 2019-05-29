# -*- coding: utf-8 -*-
# @File    : derain_pytorch/data_helper.py
# @Info    : @ TSMC-SIGGRAPH, 2019/4/16
# @Desc    : 
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -. 

import os

import torch
from PIL import Image
from matplotlib import pyplot as plt
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


# def _worker_init_fn_():
#     torch_seed = torch.initial_seed()
#     np_seed = torch_seed // (2 ** 32 - 1)
#     random.seed(torch_seed)
#     np.random.seed(np_seed)
# train_loader = DataLoader(dataset=custom_dataset, batch_size=cfg.batch_size, shuffle=True,
#                           num_workers=8, worker_init_fn=_worker_init_fn_())


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
    import numpy as np
    dataloader = get_batch()
    for i, sample in enumerate(dataloader):
        bs, ncrops, c, h, w = sample['input'].size()
        print(bs, ncrops, c, h, w)
        print(sample['input'].shape, sample['label'].shape)
        sample['input'] = sample['input'].view(-1, c, h, w)
        sample['label'] = sample['label'].view(-1, c, h, w)
        input_crops = sample['input'].numpy()
        print(np.max(input_crops), np.min(input_crops))
        label_crops = sample['label'].numpy()
        print(label_crops.shape)

        p1 = plt.subplot(121)
        p2 = plt.subplot(122)
        p1.imshow(input_crops[0, ...].transpose((1, 2, 0)))
        p2.imshow(label_crops[0, ...].transpose((1, 2, 0)))
        plt.show()
        break

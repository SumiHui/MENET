# -*- coding: utf-8 -*-
# @File    : derain_wgan_tf/transforms.py
# @Info    : @ TSMC-SIGGRAPH, 2019/5/29
# @Desc    : @ sumihui : refer to pytorch
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -. 

import numpy as np
from PIL import Image


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >> transforms.Compose([
        >>     transforms.FiveCrop(10),
        >>     lambda crops: np.stack([transforms.ToArray(crop) for crop in crops])
        >> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class FiveCrop(object):
    """Crop the given PIL Image into four corners and the central crop

    .. Note::
         This transform returns a tuple of images and there may be a mismatch in the number of
         inputs and targets your Dataset returns. See below for an example of how to deal with
         this.

    Args:
         size (sequence or int): Desired output size of the crop. If size is an ``int``
            instead of sequence like (h, w), a square crop of size (size, size) is made.
         horizontal_flip (bool): Whether use horizontal flipping or not

    Example:
         >> transform = Compose([
         >>    FiveCrop(size), # this is a list of PIL Images
         >>    lambda crops: np.stack([transforms.ToArray(crop) for crop in crops]) # returns a 4D ndarray
         >> ])
         >> #In your test loop you can do the following:
         >> input, target = batch # input is a 5d tensor, target is 2d
         >> bs, ncrops, h, w, c = input.size()
         >> result = model(input.reshape(-1, h, w, c)) # fuse batch size and ncrops
    """

    def __init__(self, size, horizontal_flip=False):
        self.size = size
        if isinstance(size, int):
            self.size = (size, size)
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size
        self.horizontal_flip = horizontal_flip

    def __call__(self, img):
        """
        :param img: (PIL Image). Image to be cropped.
        :return: return five_crop(img)
        """
        if not isinstance(img, Image.Image):
            raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

        crops = self.five_crop(img)
        if self.horizontal_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            crops = crops + self.five_crop(img)
        return crops

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

    def five_crop(self, img):
        """Crop the given PIL Image into four corners and the central crop.

        .. Note::
            This transform returns a tuple of images and there may be a
            mismatch in the number of inputs and targets your ``Dataset`` returns.

        Args:
           size (sequence or int): Desired output size of the crop. If size is an
               int instead of sequence like (h, w), a square crop (size, size) is
               made.
        Returns:
            tuple: tuple (tl, tr, bl, br, center) corresponding top left,
                top right, bottom left, bottom right and center crop.
        """
        w, h = img.size
        crop_h, crop_w = self.size
        if crop_w > w or crop_h > h:
            raise ValueError("Requested crop size {} is bigger than input size {}".format(self.size,
                                                                                          (h, w)))
        tl = img.crop((0, 0, crop_w, crop_h))
        tr = img.crop((w - crop_w, 0, w, crop_h))
        bl = img.crop((0, h - crop_h, crop_w, h))
        br = img.crop((w - crop_w, h - crop_h, w, h))
        center = self.center_crop(img)
        return (tl, tr, bl, br, center)

    def center_crop(self, img):
        """
        :param img:
        :return: PIL Image: Cropped image.
        """
        w, h = img.size
        th, tw = self.size  # Height/Width of the cropped image.
        i = int(round((h - th) / 2.))  # Upper pixel coordinate.
        j = int(round((w - tw) / 2.))  # Left pixel coordinate.
        return img.crop((j, i, j + tw, i + th))


class ToArray(object):
    """Convert a ``PIL Image`` to ``numpy.ndarray``.

    Converts a PIL Image (H x W x C) in the range
    [0, 255] to a numpy.ndarray of shape (H x W x C) in the range [0.0, 1.0].
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image): Image to be converted to numpy.ndarray.

        Returns:
            numpy.ndarray: Converted image.
        """
        return np.asarray(pic, "uint8")     # note: 2019/05/29 uint8

    def __repr__(self):
        return self.__class__.__name__ + '()'

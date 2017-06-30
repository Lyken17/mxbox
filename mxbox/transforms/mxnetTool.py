from __future__ import division

import math
import random

from PIL import Image, ImageOps
import PIL

try:
    import accimage
except ImportError:
    accimage = None

import numpy as np
import numbers
import types
import collections
import mxnet as mx


class ToNdArray(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to mx.nd.array.

    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a to mx.nd.array of shape (C x H x W) in the range [0.0, 255.0].
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = mx.nd.array((pic.transpose((2, 0, 1))))
            # backward compatibility
            return img.astype(dtype=float)

        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return mx.nd.array((nppic))

        # handle PIL Image
        if pic.mode == 'I':
            img = mx.nd.array(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = mx.nd.array(np.array(pic, np.int16, copy=False))
        else:
            # img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = mx.nd.array(np.array(pic))

        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)

        # now `img` is mx.nd.array
        img = img.reshape((pic.size[1], pic.size[0], nchannel))
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = mx.nd.transpose(img, axes=(2, 0, 1))
        # img = mx.nd.expand_dims(img, axis=0)

        return img.astype(dtype=float)


class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.

    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std

    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean, std=[1, 1, 1]):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized image.
        """
        # TODO: make efficient
        # for t, m, s in zip(tensor, self.mean, self.std):
        #     t.sub_(m).div_(s)
        for t, m, s in zip(tensor, self.mean, self.std):
            t.__isub__(m).__idiv__(s)
        return tensor
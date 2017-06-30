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

        img = img.reshape(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = mx.nd.transpose(img, axes=(2, 0, 1))

        return img.astype(dtype=float)
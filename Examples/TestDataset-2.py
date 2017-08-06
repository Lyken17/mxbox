#!/usr/bin/python
 # -*- coding: utf-8 -*-
import os.path as osp

import mxnet as mx
import mxbox
from mxbox import transforms


class SampleDst(mxbox.Dataset):
    def __init__(self, root, lst, transform=None):
        self.root = root
        with open(osp.join(root, lst), 'r') as fp:
            self.lst = [line.strip().split('\t') for line in fp.readlines()]
        self.transform = transform

    def __getitem__(self, index):
        img = self.pil_loader(osp.join(self.root, self.lst[index][0]))
        if self.transform is not None:
            img = self.transform(img)
        return {'data': img, 'softmax_label': img}

    def __len__(self):
        return len(self.lst)


if __name__ == "__main__":
    test_root = "../../Data/TestFolder/"

    trans = transforms.Compose([
        transforms.mx.ToNdArray()
    ])
    # dst = SampleDst(root=test_root, lst='train.txt', transform=trans)
    dst = mxbox.datasets.CIFAR10('../../Data/',transform=trans, download=True)

    feedin_shapes = {
        'batch_size': 2,
        'data': [mx.io.DataDesc(name='data', shape=(2, 3, 32, 32), layout='NCHW')],
        'label': [mx.io.DataDesc(name='softmax_label', shape=(2, 1), layout='N')]
    }

    loader = mxbox.DataLoader(dst, feedin_shapes, threads=1)

    for each in loader:
        print(each)
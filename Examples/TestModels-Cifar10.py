#!/usr/bin/python
# -*- coding: utf-8 -*-
import os.path as osp
import logging

logging.basicConfig(level=logging.DEBUG)

import mxnet as mx
import mxbox
from mxbox import transforms
from mxbox import models

if __name__ == "__main__":
    test_root = "../../Data/TestFolder/"

    trans = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.mx.ToNdArray()
    ])
    train_dst = mxbox.datasets.CIFAR10('../../Data/', train=True, transform=trans, download=False)
    valid_dst = mxbox.datasets.CIFAR10('../../Data/', train=False, transform=trans, download=False)

    batch_size = 32
    disp_batches = 20
    feedin_shapes = {
        'batch_size': batch_size,
        'data': [mx.io.DataDesc(name='data', shape=(batch_size, 3, 32, 32), layout='NCHW')],
        'label': [mx.io.DataDesc(name='softmax_label', shape=(batch_size,), layout='N')]
    }

    train_loader = mxbox.DataLoader(train_dst, feedin_shapes, threads=1)
    valid_loader = mxbox.DataLoader(valid_dst, feedin_shapes, threads=1)

    sym = models.resnet_getsym(num_classes=10, num_layers=20, image_shape='3,32,32')

    # devices for training
    devs = mx.gpu(0)

    # create model
    model = mx.mod.Module(
        context=devs,
        symbol=sym
    )

    optimizer_params = {
        'learning_rate': 0.1,
        'momentum': 0.9,
        'wd': 2e-5, }

    # evaluation metrices
    eval_metrics = ['accuracy']

    # callbacks that run after each batch
    batch_end_callbacks = [mx.callback.Speedometer(batch_size, frequent=disp_batches)]

    # run
    model.fit(train_loader,
              begin_epoch=0,
              num_epoch=30,
              eval_data=valid_loader,
              eval_metric=eval_metrics,
              kvstore="device",
              optimizer="sgd",
              optimizer_params=optimizer_params,
              initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
              batch_end_callback=batch_end_callbacks,
              allow_missing=True)
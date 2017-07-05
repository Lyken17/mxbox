MXbox: The missing toolbox for mxnet.
=====================================

MXbox is a toolbox aiming to provide a general and simple interface for vision tasks. This project is greatly inspired by PyTorch_ and torchvision_. Detailed copyright file will be attached later. Improvements and suggestions are welcome.

.. _PyTorch: https://github.com/pytorch/pytorch
.. _torchvision: https://github.com/pytorch/vision

Installation
============
.. code:: bash

    pip install mxbox

Features
========
1) Define preprocess on the fly

.. code:: python

    transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.mx.ToNdArray(),
        transforms.mx.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                                std = [ 0.229, 0.224, 0.225 ]),
    ])

PS: By default, mxbox use PIL_ to read and transform images. But it also supports other backends like skimage_ and Numpy_.

Documentation
=============
Under construction, coming soon.

TODO list
=========

1) Random shuffle

2) Efficient multi thread reading

3) Common Models
MXbox: Simple, efficient and flexible vision toolbox for mxnet framework.
========================================================================

MXbox is a toolbox aiming to provide a general and simple interface for vision tasks. This project is greatly inspired by PyTorch_ and torchvision_. Detailed copyright files will be attached later. Improvements and suggestions are welcome.

.. _PyTorch: https://github.com/pytorch/pytorch
.. _torchvision: https://github.com/pytorch/vision

Installation
============
.. code:: bash

    pip install mxbox

Features
========
1) Define **preprocess** as a flow

.. code:: python

    transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.mx.ToNdArray(),
        transforms.mx.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                                std  = [ 0.229, 0.224, 0.225 ]),
    ])

PS: By default, mxbox uses `PIL` to read and transform images. But it also supports other backends like `accimage` and `skimage`.

More examples can be found in XXX.

2) Build **DataLoader** in several lines

.. code:: python

    feedin_shapes = {
        'batch_size': 8,
        'data': [mx.io.DataDesc(name='data', shape=(8, 3, 32, 32), layout='NCHW')],
        'label': [mx.io.DataDesc(name='softmax_label', shape=(8, 1), layout='N')]
    }

    dst = Dataset(root='../../data', transform=img_transform, label_transform=label_transform)
    loader = DataLoader(dst, feedin_shapes, threads=1, shuffle=True)

Also, common datasets such as `cifar10`, `cifar100`, `SVHN`, `MNIST` are out-of-the-box.

3) Load popular model and pretrained weights

.. code:: python

    vgg = mxbox.models.vgg(num_classes=10, pretrained=True)
    resnet = mxbox.models.resnet152(num_classes=10, pretrained=True)


Documentation
=============
Under construction, coming soon.


TODO list
=========

1) Efficient multi-thread reading (Prefetch wanted

2) Common Models preparation.

3) More friendly error information.
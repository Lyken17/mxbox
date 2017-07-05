MXbox: The missing toolbox for mxnet.
=====================================

MXbox is a toolbox aiming to provide a general and simple interface for vision tasks. This project is greatly inspired by PyTorch_ and torchvision_. Detailed copyright files will be attached later. Improvements and suggestions are welcome.

.. _PyTorch: https://github.com/pytorch/pytorch
.. _torchvision: https://github.com/pytorch/vision

Installation
============
.. code:: bash

    pip install mxbox

Features
========
1) Define **preprocess** in a flow

.. code:: python

    transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.mx.ToNdArray(),
        transforms.mx.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                                std  = [ 0.229, 0.224, 0.225 ]),
    ])

PS: By default, mxbox use PIL to read and transform images. But it also supports other backends like skimage_and Numpy.

More examples can be found in XXX.

2) Build **DataLoader** in twenty lines

.. code:: python

    class TestDataset(Dataset):
        def __init__(self, root="../../data", transform=None, label_transform=None):
            super(TestDataset, self).__init__()
            self.root = root

            self.transform = transform
            self.label_transform = label_transform

            with open(osp.join(root, 'caltech-256-60-train.lst'), 'r') as fp:
                self.flist = [line.strip().split() for line in fp.readlines()]

        def __getitem__(self, index):
            label, id, path, = self.flist[index]
            data = self.pil_loader(osp.join(self.root, '256_ObjectCategories', path))

            if self.transform is not None:
                data = self.transform(data)
            if self.label_transform is not None:
                label = self.label_transform(label)

            return [data], [label]

        def __len__(self):
            return len(self.flist)

    feedin_shapes = {
        'batch_size': 8,
        'data': [mx.io.DataDesc(name='data', shape=(32, 3, 128, 128), layout='NCHW')],
        'label': [mx.io.DataDesc(name='softmax_label', shape=(32,), layout='N')]
    }

    dst = TestDataset(root='../../data', transform=img_transform, label_transform=label_transform)
    loader = BoxLoader(dst, feedin_shapes, collate_fn=mx_collate, num_workers=1)

3) Load popular model and pretrained weights

Coming soon


Documentation
=============
Under construction, coming soon.

TODO list
=========

1) Random shuffle

2) Efficient multi thread reading

3) Common Models
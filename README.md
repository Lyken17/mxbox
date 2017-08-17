# MXbox: Simple, efficient and flexible vision toolbox for mxnet framework.

MXbox is a toolbox aiming to provide a general and simple interface for vision tasks. This project is greatly inspired by [PyTorch](https://github.com/pytorch/pytorch) and [torchvision](https://github.com/pytorch/vision). Detailed copyright files are on the way. Improvements and suggestions are welcome.


## Installation
```bash
pip install mxbox
```

## Features
1. Define **preprocess** as a flow

```python
transform = transforms.Compose([
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.mx.ToNdArray(),
    transforms.mx.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                            std  = [ 0.229, 0.224, 0.225 ]),
])
```

PS: By default, mxbox uses `PIL` to read and transform images. But it also supports other backends like `accimage` and `skimage`.

More usages can be found in [documents](mxbox/transforms/README.md) and [examples](Examples/).

2) Build an multi-thread **DataLoader** in few lines

Common datasets such as `cifar10`, `cifar100`, `SVHN`, `MNIST` are out-of-the-box. You can simply load them from `mxbox.datasets`.

```python
from mxbox import transforms, datasets, DataLoader
trans = transforms.Compose([
        transforms.mx.ToNdArray(), 
        transforms.mx.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                                std  = [ 0.229, 0.224, 0.225 ]),
])
dataset = datasets.CIFAR10('~/.mxbox/cifar10', transform=trans, download=True)

batch_size = 32
feedin_shapes = {
    'batch_size': batch_size,
    'data': [mx.io.DataDesc(name='data', shape=(batch_size, 3, 32, 32), layout='NCHW')],
    'label': [mx.io.DataDesc(name='softmax_label', shape=(batch_size, ), layout='N')]
}
loader = DataLoader(dataset, feedin_shapes, threads=8, shuffle=True)
```  

Or you can also easily create your own, which only requires to implement `__getitem__` and `__len__`.

```python
class TooYoungScape(mxbox.Dataset):
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
        
dataset = TooYoungScape('~/.mxbox/TooYoungScape', "train.lst", transform=trans)
loader = DataLoader(dataset, feedin_shapes, threads=8, shuffle=True)
```
    

3) Load popular model with pretrained weights

Note: current under construction, many models lack of pretrained weights and some of their definition files are missing.


```python
vgg = mxbox.models.vgg(num_classes=10, pretrained=True)
resnet = mxbox.models.resnet152(num_classes=10, pretrained=True)
```

## TODO list

0) FLAG options? 

1) Efficient multi-thread reading (Prefetch wanted

2) Common Models preparation.

3) More friendly error logging.
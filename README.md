# Multi-GPU training with Pytorch.

Let's try easy multi-GPU training with pytorch.

Just adding:

```
if device == 'cuda':
    net = torch.nn.DataParallel(net) # make parallel
    cudnn.benchmark = True
```
will enable parallel GPU usage in Pytorch! :)

Training dataset: CIFAR10

DeepLearningFramework: Pytorch

Should scale more than 2 GPUs!

## 詳しい解説と設定など
https://qiita.com/arutema47/items/2b92f94c734b0a11609d

## Usage

```
git clone https://github.com/kentaroy47/pytorch-mgpu-cifar10.git
cd pytorch-mgpu-cifar10
export CUDA_VISIBLE_DEVICES=0,1

python train_cifar10.py
```

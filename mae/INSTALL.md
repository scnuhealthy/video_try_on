# Installation

## Requirements
- Python >= 3.8
- Numpy
- PyTorch >= 1.3
- [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  You can install them together at [pytorch.org](https://pytorch.org) to make sure of this.
- simplejson: `pip install simplejson`
- GCC >= 4.9
- PyAV: `conda install av -c conda-forge`
- ffmpeg (4.0 is prefereed, will be installed along with PyAV)
- tqdm: `pip install tqdm`
- iopath: `pip install -U iopath` or `conda install -c iopath iopath`
- psutil: `pip install psutil`
- OpenCV: `pip install opencv-python`
- torchvision: `pip install torchvision` or `conda install torchvision -c pytorch`
- tensorboard: `pip install tensorboard`
- timm: `pip install timm==0.3.2`

## Pytorch
Please follow PyTorch official instructions to install from source:
```
git clone --recursive https://github.com/pytorch/pytorch
```

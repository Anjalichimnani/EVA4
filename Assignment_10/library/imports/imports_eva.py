from __future__ import print_function, with_statement, division

import torch

from torchvision import models
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import _LRScheduler
import torch.optim as optim

from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

import albumentations as A

from albumentations import Compose, RandomCrop, Normalize, HorizontalFlip, Resize, HueSaturationValue, Rotate, RGBShift, Cutout
from albumentations.pytorch import ToTensor

from datetime import datetime

from PIL import Image
import cv2 

import urllib.request
import copy
import os
import os.path as osp

import click
import matplotlib.cm as cm
import matplotlib.pyplot as plt

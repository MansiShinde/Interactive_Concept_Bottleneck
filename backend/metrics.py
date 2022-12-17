from os.path import isfile, isdir, join
from collections import defaultdict as ddict
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models
from tqdm.notebook import tqdm
from os import listdir
import torch.nn as nn
from PIL import Image
import numpy as np
import torch
import pickle


def accuracy(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k
    output and target are Torch tensors
    """
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    temp = target.view(1, -1).expand_as(pred)
    correct = pred.eq(temp)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def binary_accuracy(output, target):
    """
    Computes the accuracy for multiple binary predictions
    output and target are Torch tensors
    """
    pred = output.cpu() >= 0.5

    acc = (pred.int()).eq(target.int()).sum()
    acc = acc*100 / np.prod(np.array(target.size()))
    return acc
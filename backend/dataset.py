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
from configs import N_ATTRIBUTES, N_CLASSES, batch_size

class CUBDataset():
    def __init__(self, pkl_file_path, use_attr, is_train, no_img, uncertain_label,image_dir, n_class_attr, transform=None):
        """
        Arguments:
        pkl_file_paths: list of full path to all the pkl data
        use_attr: whether to load the attributes (e.g. False for simple finetune)
        no_img: whether to load the images (e.g. False for A -> Y model)
        uncertain_label: if True, use 'uncertain_attribute_label' field (i.e. label weighted by uncertainty score, e.g. 1 & 3(probably) -> 0.75)
        image_dir: default = 'images'. Will be append to the parent dir
        n_class_attr: number of classes to predict for each attribute. If 3, then make a separate class for not visible
        transform: whether to apply any special transformation. Default = None, i.e. use standard ImageNet preprocessing
        """
        self.data = []

        # self.is_train = any(["train" in path for path in pkl_file_path])
        # if not self.is_train:
        #     assert any([("test" in path) or ("val" in path) for path in pkl_file_path])

        # for file_path in pkl_file_path:
        #     self.data.extend(pickle.load(open(file_path, 'rb')))

        self.data.extend(pickle.load(open(pkl_file_path, 'rb')))
        self.is_train = is_train
        self.transform = transform
        self.use_attr = use_attr
        self.no_img = no_img
        self.uncertain_label = uncertain_label
        self.image_dir = image_dir
        self.n_class_attr = n_class_attr

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_data = self.data[idx]
        img_path = img_data['img_path']
        # Trim unnecessary paths
        try:
            idx = img_path.split('/').index('CUB_200_2011')
            if self.image_dir != 'images':
                img_path = '/'.join([self.image_dir] + img_path.split('/')[idx+1:])
                img_path = img_path.replace('images/', '')
            else:
                img_path = '/'.join(img_path.split('/')[idx:])
            img = Image.open(img_path).convert('RGB')
        except:
            img_path_split = img_path.split('/')
            split = 'train' if self.is_train else 'test'
            img_path = '/'.join(img_path_split[:2] + [split] + img_path_split[2:])
            img = Image.open(img_path).convert('RGB')

        class_label = img_data['class_label']
        if self.transform:
            img = self.transform(img)

        if self.use_attr:
            if self.uncertain_label:
                attr_label = img_data['uncertain_attribute_label']
            else:
                attr_label = img_data['attribute_label']
            if self.no_img:
                if self.n_class_attr == 3:
                    one_hot_attr_label = np.zeros((N_ATTRIBUTES, self.n_class_attr))
                    one_hot_attr_label[np.arange(N_ATTRIBUTES), attr_label] = 1
                    return one_hot_attr_label, class_label
                else:
                    return attr_label, class_label
            else:
                return img, class_label, attr_label
        else:
            return img, class_label




def load_data(pkl_file_path, use_attr, is_train, no_img, uncertain_labels, image_dir, n_class_attr):

    transform = transforms.Compose([
    transforms.RandomResizedCrop(299),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), #implicitly divides by 255
    transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
    #transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ]),
    ])

    if no_img: ## if no_img is true, then call C->Y 
        dataset = CUBDataset(pkl_file_path, use_attr, is_train, no_img, uncertain_labels, image_dir, n_class_attr, transform)
    else: ## if no_img is false, then call X-> C
        dataset = CUBDataset( pkl_file_path, use_attr, is_train, no_img, uncertain_labels, image_dir, n_class_attr, transform)

    loader = DataLoader(dataset, batch_size,shuffle=True,num_workers=8)

    return loader
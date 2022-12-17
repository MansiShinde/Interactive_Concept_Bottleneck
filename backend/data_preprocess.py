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


data_path = "/content/CUB_200_2011/images"

path_to_id_map = dict()


with open(data_path.replace('images', 'images.txt'), 'r') as f:
    for line in f:
        items = line.strip().split()
        path_to_id_map[join(data_path, items[1])] = int(items[0])


attribute_labels_all = ddict(list) #map from image id to a list of attribute labels
attribute_certainties_all = ddict(list) #map from image id to a list of attribute certainties
attribute_uncertain_labels_all = ddict(list) #map from image id to a list of attribute labels calibrated for uncertainty

uncertainty_map = {1: {1: 0, 2: 0.5, 3: 0.75, 4:1}, #calibrate main label based on uncertainty label
                    0: {1: 0, 2: 0.5, 3: 0.25, 4: 0}}


with open('../CUB_200_2011/attributes/image_attribute_labels.txt', 'r') as f:
    for line in f:
        file_idx, attribute_idx, attribute_label, attribute_certainty = line.strip().split()[:4]
        attribute_label = int(attribute_label)
        attribute_certainty = int(attribute_certainty)
        uncertain_label = uncertainty_map[attribute_label][attribute_certainty]
        attribute_labels_all[int(file_idx)].append(attribute_label)
        attribute_uncertain_labels_all[int(file_idx)].append(uncertain_label)
        attribute_certainties_all[int(file_idx)].append(attribute_certainty)


is_train_test = dict() #map from image id to 0 / 1 (1 = train)
with open('../CUB_200_2011/train_test_split.txt', 'r') as f:
    for line in f:
        idx, is_train = line.strip().split()
        is_train_test[int(idx)] = int(is_train)
print("Number of train images from official train test split:", sum(list(is_train_test.values())))


train_data, test_data = [], []
folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]
folder_list.sort() #sort by class index
for i, folder in enumerate(folder_list):
  folder_path = join(data_path, folder)
  classfile_list = [cf for cf in listdir(folder_path) if (isfile(join(folder_path,cf)) and cf[0] != '.')]
  #classfile_list.sort()
  for cf in classfile_list:
      img_id = path_to_id_map[join(folder_path, cf)]
      img_path = join(folder_path, cf)
      metadata = {'id': img_id, 'img_path': img_path, 'class_label': i,
                'attribute_label': attribute_labels_all[img_id], 'attribute_certainty': attribute_certainties_all[img_id],
                'uncertain_attribute_label': attribute_uncertain_labels_all[img_id]}
      if is_train_test[img_id]:
          train_data.append(metadata)
      else:
          test_data.append(metadata)

pickle_train = open("../CUB_200_2011/train_data.pickle","wb")
pickle.dump(train_data, pickle_train)
pickle_train.close()

pickle_test = open("../CUB_200_2011/test_data.pickle","wb")
pickle.dump(test_data, pickle_test)
pickle_test.close()

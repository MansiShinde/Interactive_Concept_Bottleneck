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
import math
from configs import N_ATTRIBUTES, N_CLASSES, batch_size, num_epochs
from models import MultiLayerPerceptron
from dataset import load_data
from metrics import accuracy, binary_accuracy
from load_model import *
import torch.nn.functional as F



con_model = define_con_model()
svm_model = define_mlp_model()


loader = load_data(pkl_file_path="../CUB_200_2011/test_data.pickle", use_attr= True, is_train= False, no_img=False, uncertain_labels=True, image_dir='images', n_class_attr=2)

use_attr = True
no_img = False
attr_loss_weight = 1
bottleneck = True
criterion = torch.nn.CrossEntropyLoss()
attr_criterion = [] #separate criterion (loss function) for each attribute

con_model.eval()
svm_model.eval()

acc_meter = 0
num_epochs = 5

for epoch in range(num_epochs):

    for _, data in enumerate(tqdm(loader, desc="Testing", leave=False)):
        if attr_criterion is None:
            inputs, labels = data
            attr_labels, attr_labels_var = None, None
        else:
            inputs, labels, attr_labels = data
            if N_ATTRIBUTES > 1:
                attr_labels = [i.long() for i in attr_labels]
                attr_labels = torch.stack(attr_labels).t()#.float() #N x 312
            else:
                if isinstance(attr_labels, list):
                    attr_labels = attr_labels[0]
                attr_labels = attr_labels.unsqueeze(1)
            attr_labels_var = torch.autograd.Variable(attr_labels).float()
            # attr_labels_var = attr_labels_var.cuda() if torch.cuda.is_available() else attr_labels_var

        inputs_var = torch.autograd.Variable(inputs)
        # inputs_var = inputs_var.cuda() if torch.cuda.is_available() else inputs_var
        labels_var = torch.autograd.Variable(labels)
        # labels_var = labels_var.cuda() if torch.cuda.is_available() else labels_var

        concepts = con_model(inputs_var)  ##predict concepts 

        concept_prob = F.softmax(concepts, dim=1) ## store the values between 0 and 1 
        concepts_0_1 = (concept_prob>0.5).float() ## convert the concept values to 0 and 1

        concepts_0_1 = concepts_0_1.tolist()
        concepts_0_1 = [[math.ceil(concepts_0_1) for concepts_0_1 in concepts_0_1[0]]]

        svm_pred = svm_model.predict(concepts_0_1)
        svm_pred =  svm_pred.tolist()
        x = torch.FloatTensor(svm_pred).view(torch.FloatTensor(svm_pred).size(0), -1)
        acc_meter = accuracy( x[0], labels, topk=(1,))


        print("Epoch: {}/{} — Accuracy: {:.4f}".format(epoch+1, num_epochs, acc_meter))


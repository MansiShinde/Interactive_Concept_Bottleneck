
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
from configs import N_ATTRIBUTES, N_CLASSES, batch_size, num_epochs
from models import con_model, MultiLayerPerceptron
from dataset import load_data
from metrics import accuracy, binary_accuracy
import sys
from sklearn.svm import SVC
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

def sequential():

    con_model = models.resnet18(pretrained=True)

    con_model.fc = nn.Sequential(
        nn.Linear(con_model.fc.in_features, 312)
    )

    svm_model = SVC("C": 100 ,kernel='rbf',"gamma": 0.001 )

    use_attr = True
    no_img = False
    attr_loss_weight = 1
    bottleneck = True
    criterion = torch.nn.CrossEntropyLoss()
    attr_criterion = [] #separate criterion (loss function) for each attribute


    if use_attr and not no_img:
        for i in range(N_ATTRIBUTES):
            attr_criterion.append(torch.nn.CrossEntropyLoss())
    else:
        attr_criterion = None


    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, svm_model.parameters()), lr=0.01, weight_decay=0.00004)

    loader = load_data(pkl_file_path="../CUB_200_2011/train_data.pickle", use_attr=use_attr, is_train=True, no_img=False, uncertain_labels=True, image_dir="images", n_class_attr=2)
    ## training the model 


    loss_meter = 0
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

            concepts = con_model(inputs_var) 

            concept_prob = F.softmax(concepts, dim=1) ## store the values between 0 and 1 
            concepts_0_1 = (concept_prob>0.5).float() ## convert the concept values to 0 and 1

            concepts_0_1 = concepts_0_1.tolist()
            concepts_0_1 = [[math.ceil(concepts_0_1) for concepts_0_1 in concepts_0_1[0]]]
            
            svm_pred = svm_model.fit(concepts_0_1)

            loss = criterion(torch.Tensor(svm_pred[0]), labels_var)

            loss_meter = loss.item()

            optimizer.zero_grad() #zero the parameter gradients
            loss.backward()
            optimizer.step() #optimizer step to update parameters

            svm_accuracy = accuracy_score(labels.tolist(), svm_pred)

            print("Loss:",loss_meter.item())
            print("Accuracy:", svm_accuracy)
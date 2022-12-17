import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
import numpy as np
from load_model import *
import torch.nn.functional as F
import math 
import re


def predict_bird(image_path):

    transform = transforms.Compose([
    transforms.RandomResizedCrop(299),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), #implicitly divides by 255
    transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
    ])

    img = Image.open(image_path).convert('RGB')
    input = transform(img)

    input =input.unsqueeze(0)

    con_model = define_con_model()
    svm_model = define_mlp_model()

    concepts = con_model(input)
    concept_prob = F.softmax(concepts, dim=1) ## store the values between 0 and 1 
    concepts_0_1 = (concept_prob>0.5).float() ## convert the concept values to 0 and 1

    concepts_0_1 = concepts_0_1.tolist()
    concepts_0_1 = [[math.ceil(concepts_0_1) for concepts_0_1 in concepts_0_1[0]]]
    
    probability = svm_model.predict_proba(concepts_0_1)

    attributes = []
    with open("../CUB_200_2011/attributes.txt", 'r') as f:
        for line in f:
            items = line.strip().split()
            attributes.append(items[1])

### data to send to UI
    attribute_prob, classes_prob = send_data_to_UI(concept_prob.tolist()[0], probability[0], attributes)

    return attribute_prob, classes_prob


def rerun_class(concept_prob_dict):
    attributes = []
    concept_prob = []
    for k, v in concept_prob_dict.items():
        attributes.append(k)
        concept_prob.append(float(v))

    svm_model = define_mlp_model() ##svm model

    concepts_0_1 = [(np.array(concept_prob)>0.5).astype(int).tolist()]

    probability = svm_model.predict_proba(concepts_0_1)
    
    attribute_prob, classes_prob = send_data_to_UI(concept_prob, probability[0], attributes)

    return attribute_prob, classes_prob


def send_data_to_UI(concept_prob, output, attributes):

    attribute_prob = dict()
    for i in range(0, len(attributes)):
        attribute_prob[attributes[i]] = concept_prob[i]

    class_names = []
    with open("../CUB_200_2011/CUB_200_2011/classes.txt", 'r') as f:
        for line in f:
            res = re.search("\.(.*)", line)
            class_names.append(res.groups()[0])
     
    classes_prob = dict()
    for i in range(0,len(class_names)):
        classes_prob[class_names[i]] = output[i]

    return attribute_prob, classes_prob 


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


def train_concept():

    con_model = models.resnet18(pretrained=True)

    con_model.fc = nn.Sequential(
        nn.Linear(con_model.fc.in_features, 312)
    )

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


    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01, weight_decay=0.00004)

    loader = load_data(pkl_file_path="../CUB_200_2011/train_data.pickle", use_attr=use_attr, is_train=True, no_img=False, uncertain_labels=True, image_dir="images", n_class_attr=2)
    ## training the model 


    loss_meter = 0
    acc_meter = 0

    con_model.train()


    for epoch in range(num_epochs):
    
        for _, data in enumerate(tqdm(loader, desc="Training", leave=False)):
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
            print("inputs_var:",inputs_var.data.cpu().numpy().shape)
            # inputs_var = inputs_var.cuda() if torch.cuda.is_available() else inputs_var
            labels_var = torch.autograd.Variable(labels)
            # labels_var = labels_var.cuda() if torch.cuda.is_available() else labels_var
            print("labels_var:",labels_var.data.cpu().numpy().shape)

            print("attr_labels_var:",attr_labels_var.data.cpu().numpy().shape)

            outputs = con_model(inputs_var)
            print("outputs:",outputs.data.cpu().numpy().shape)

            losses = []

            if not bottleneck:
                loss_main = criterion(outputs[0], labels_var)
                losses.append(loss_main)

            if attr_criterion is not None and attr_loss_weight > 0: #X -> A, cotraining, end2end
                for i in range(len(attr_criterion)):
                    loss_main = attr_criterion[i](outputs[:,i], attr_labels_var[:, i])
                    losses.append(attr_loss_weight * loss_main)

            if bottleneck: #attribute accuracy
                sigmoid_outputs = torch.nn.Sigmoid()(outputs)
                acc = binary_accuracy(sigmoid_outputs, attr_labels)
                acc_meter = acc.data.cpu().numpy()


            if attr_criterion is not None:
                if bottleneck:
                    total_loss = sum(losses)/ N_ATTRIBUTES
                else: 
                    total_loss = losses[0] + sum(losses[1:])
            
            loss_meter = total_loss.item()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            print("Epoch: {}/{} — Loss: {:.4f}".format(epoch+1, num_epochs, loss_meter))
            print("Accuracy:", acc_meter)


    model_scripted = torch.jit.script(con_model) # Export to TorchScript
    model_scripted.save('backend/models/model_scripted.pt')


def train_classifier():

    train_data = []
    train_data.extend(pickle.load(open("/content/train_data.pickle", 'rb')))

    test_data = []
    test_data.extend(pickle.load(open("/content/test_data.pickle", 'rb')))

    test_attribute_labels = []
    test_class_labels = []
    for i in range(0, len(test_data)):
        test_attribute_labels.append(test_data[i]["attribute_label"])
        test_class_labels.append(test_data[i]['class_label'])

    attribute_labels = []
    class_labels = []
    for i in range(0, len(train_data)):
        attribute_labels.append(train_data[i]["attribute_label"])
        class_labels.append(train_data[i]['class_label'])


    params_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [[1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    svc= SVC(probability=True)
    # Performing CV to tune parameters for best SVM fit 
    svm_model = GridSearchCV(svc, params_grid, cv=5)
    svm_model.fit(attribute_labels, class_labels)

    print('Best score for training data:', svm_model.best_score_,"\n") 

    # View the best parameters for the model found using grid search
    print('Best C:',svm_model.best_estimator_.C,"\n") 
    print('Best Kernel:',svm_model.best_estimator_.kernel,"\n")
    print('Best Gamma:',svm_model.best_estimator_.gamma,"\n")

    final_model = svm_model.best_estimator_
    Y_pred = final_model.predict_proba(test_attribute_labels)

    probability = final_model.predict_proba(test_attribute_labels)


    pickle.dump(final_model, open("backend/models/model_scripted_svm.sav", 'wb'))

if __name__ == '__main__':
    if sys.argv[0] == "concept" :
        train_concept()
    elif sys.argv[0] == "classifier":
        train_classifier()

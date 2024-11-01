#######################################################################################################
#### Script for training a pytorch, convolutional neural net, using the pre-trained resnet18 model ####
#### Authors:  Hieu Le & Grant Humphries
#### Date: August 2018
#### This script was written for the Spacewhale project 
#######################################################################################################
#### Usage examples (Linux)
####
####  $python training_script.py --name MODEL1 --model resnet18 --data_dir /home/ghumphries/spacewhale/data --verbose True --epochs 19
####
#######################################################################################################
#### Setup information
####    To run this script, ensure that you have your training images inside of a folder called 'train'.
####    Inside of the train folder, your images must be organized into folders based on the label.  For example: 
####    ./train/Water - this folder contains only water images in .png format
####    ./train/Whale - this folder contains only whale images in .png format
####    IMPORTANT:
####        The --data_dir argument must point to the folder ABOVE the 'train' folder. For example:
####        .home/user/spacewhale/fulldata/train/... ->  data_dir usage:  --data_dir /home/user/spacewhale/fulldata
####
#######################################################################################################
### Library imports
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from m_util2 import *
import argparse
from model import define_model
#######################################################################################################

### Create arguments for command line interface
parser = argparse.ArgumentParser()
parser.add_argument('--name',type=str)
parser.add_argument('--model',type=str)
parser.add_argument('--data_dir',type=str)
parser.add_argument('--verbose',type=bool,default=False)
parser.add_argument('--epochs',type=int,default=25)
parser.add_argument('--lr', type=float)

opt = parser.parse_args()

### Load the spacewhale class
s = spacewhale(opt)

### This creates a new directory called 'trained model' in the directory you are currently working from in Terminal
opt.checkpoint = ('./trained_model/'+opt.name)
s.sdmkdir(opt.checkpoint)

#Preparing the data
data_dir = opt.data_dir
print('######################################################################################################')
print('WELCOME TO SPACEWHALE!')
print('######################################################################################################')
print('We will now train your model.. please be patient')
print('Using', opt.model, 'Your trained model will be named', opt.name) 
print('------------------------------------------------------------------------------')

### This part loads up any folders in the 'train' folder with the label being the name of the folder
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir,x),s.data_transforms[x]) for x in ['train']}
#image_datasets = datasets.ImageFolder(data_dir)
#print(len(image_datasets.classes))
### Now we load training data, but we use a weighted sampler because there are more water images than whales
#weights = s.make_weights_for_balanced_classes(image_datasets.imgs, len(image_datasets.classes)) #2= num. of classes
weights = s.make_weights_for_balanced_classes(image_datasets['train'].imgs, len(image_datasets['train'].classes)) #2= num. of classes
#print(weights)
weights = torch.DoubleTensor(weights)                                       
#print('torch.DoubleTensor')
#print(weights)
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
#print(len(image_datasets.imgs))
#print(weights)
#dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=4, sampler = sampler, num_workers=4)
dataloaders = torch.utils.data.DataLoader(image_datasets['train'], batch_size=32, sampler = sampler, num_workers=4, drop_last = True)

#dataloaders = torch.utils.data.DataLoader
print(dataloaders)

dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}
#dataset_sizes = len(image_datasets.classes)

print('Your dataset size is: %d'%(dataset_sizes['train']))
#class_names = image_datasets.classes
class_names = image_datasets['train'].classes
print('You have',str(len(class_names)),'classes in your dataset')

print('------------------------------------------------------------------------------')
print('Labels for the dataset are:')
print(class_names[0] + ' = 0')
print(class_names[1] + ' = 1')
print('------------------------------------------------------------------------------')
### This sets the device (if cuda is installed properly, it will send the training data to the gpu)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dev = ("gpu" if torch.cuda.is_available() else "cpu")
print('Data loaded into', dev)
print('------------------------------------------------------------------------------')

######################################################################################################################
### This part defines the model we're going to use
### First it downloads the pretrained resnet model (if we dont' have it) from modelZoo
### We count the number of features in the model and then replace the last layer with a linear layer so we can map our own classes
### The model is sent to the gpu and we then opt to use CrossEntropy as the loss function
### The optimizer is set as stochastic gradient descent with a learning rate of 0.001
### We then set the learning rate to decay every 7 epochs 

### Here we can put in other available models
model_ft = define_model(name = opt.model)
model_ft = model_ft.to(device)


criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=opt.lr, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

### If the verbose option is set, then print out the model
if opt.verbose:
    print(model_ft)

######################################################################################################################
### Run the train_model function from the spacewhale class
print("Training Model: Learning Rate = ", opt.lr)
print("Model type: ", opt.model)

model_ft = s.train_model(opt, device, dataset_sizes, dataloaders, model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=opt.epochs)

######################################################################################################################

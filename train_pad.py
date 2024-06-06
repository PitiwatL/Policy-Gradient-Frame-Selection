import numpy as np # linear algebra
import pandas as pd

from torchvision import transforms

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.models as models
# from torchvision.models import vgg16

import torch.nn.functional as F
from torchvision.utils import make_grid

import cv2
import os, shutil
from PIL import Image
# import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import train_test_split

import tqdm
from tqdm import tqdm
import time

# # From our class
from Utils.ImageDataset import ImageDataset1, ImageDataset2, ImageDatasetPad
from Utils.skim_net import SkimNetworkPad, EvalNetwork, EvalNetworkPad
from Utils.Losses import FocalLoss

from Utils.LSTM import LSTMCells


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.cuda.is_available()
print("GPU Available: ", x)

Target_PATH = "MiniDataFrame"

Class = ['A001', 'A002', 'A003', 'A004', 'A005']

Class = sorted(Class)
INDEX = [i for i in range(60)]

X_train = []
X_val = []
y_train = []
y_val = []
# Datasett = []
# Labels = []

### Cross Subject
##### Train on the fly
for path in os.listdir(Target_PATH):
    if Class.index(path[16:20]) in INDEX:
        if path[8:12] in ['P001', 'P002', 'P004', 'P005', 'P008', 'P009', 'P013', 'P014', 'P015', 
                            'P016', 'P017', 'P018', 'P019', 'P025', 'P027', 'P028', 'P031', 'P034', 
                            'P035', 'P038'] :
            
            X_train.append(Target_PATH + "/" + path)
            y_train.append(Class.index(path[16:20]))

for path in os.listdir(Target_PATH):
    if Class.index(path[16:20]) in INDEX:
        if path[8:12] in ['P003', 'P006', 'P007', 'P010', 'P011', 'P012', 'P020', 'P021', 'P022',
                            'P023', 'P024', 'P026', 'P029', 'P030', 'P032', 'P033', 'P036', 'P037', 
                            'P039', 'P040'] :
            
            X_val.append(Target_PATH + "/" + path)
            y_val.append(Class.index(path[16:20]))

print('Num train: ', len(y_train))
print('Num test: ', len(y_val))

frame_size = 224

train_transform = transforms.Compose([
        transforms.Resize((frame_size, frame_size)),
        # transforms.CenterCrop(224),          ##########################
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    
train_dataset = ImageDatasetPad(data_paths = X_train,
                             labels     = y_train,
                             transform  = train_transform,
                             size = frame_size, 
                             target_num_frame = 5, 
                             mode = 'train' )

val_dataset   = ImageDatasetPad(data_paths = X_val, 
                             labels     = y_val,
                             transform  = train_transform,
                             size = frame_size, 
                             target_num_frame = 5, mode = 'test')

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False)

# #### Define the based model
# model_base = SkimNetworkPad(128)
# model = model_base.to(device)
# model = torch.nn.DataParallel(model)

model_base = EvalNetworkPad(128)
model = model_base.to(device)
model = torch.nn.DataParallel(model)

# model.load_state_dict(torch.load('/project/lt200210-action/OCS_Sampler/weights/Skim_ResNet18_LSTM_224_20.pt'))
# print("Load weights complete!")

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.6, 
                                     patience=3, min_lr= 1 * 1e-6 ,verbose = True)

# ########################
# ###### load Weight #####
# ########################
# # model.load_state_dict(torch.load('/project/lt200210-action/OCS_Sampler/weights/Eval_ResNet18_LSTM_Skim_224_5.pt'))

#### Start Training Loop
epochs = 200

val_loss_his = []
train_loss_his = []
stop = 0
for eph in range(epochs):
    train_preds = []
    train_target = []
    loss_epoch_train=[]
    loss_epoch_val = []
    
    Train_Correct = 0
    Num_Train = 0
    
    # Run the training batches
    model.train()
    for b, (X_train, y_train) in tqdm(enumerate(train_loader), total=len(train_loader)):
        optimizer.zero_grad()
        
        X_train, y_train = X_train.to(device), y_train.to(device)
        
        output  = model(X_train)
        y_pred = output
        y_prd  = torch.argmax(y_pred, 1)
        
        Train_Correct += torch.sum(y_prd == y_train)
        
        Num_Train += y_train.shape[0]
        
        loss = criterion(y_pred.cpu(), y_train.cpu())
        
        loss_epoch_train.append(loss.item())
                
        loss.backward()
        optimizer.step()
        
    
    train_loss_his.append(np.mean(loss_epoch_train))
    train_acc = Train_Correct/Num_Train
    print(f'epoch: {eph:2}  Train loss: {np.mean(loss_epoch_train):10.7f} : Train Acc {train_acc:10.7f}')
    
    # Run the validation batches
    Val_Correct = 0
    Num_Val = 0
    model.eval()
    with torch.no_grad():
        for b, (X_val, y_val) in enumerate(val_loader):
            X_val, y_val = X_val.to(device), y_val.to(device)
            
            out_val = model(X_val)
            
            y_val_ = torch.argmax(out_val, 1)
            Val_Correct += torch.sum(y_val_ == y_val)
            
            Num_Val += y_val.shape[0]
            loss = criterion(out_val.cpu(), 
                             y_val.cpu())
            
            loss_epoch_val.append(loss.item())
            
    val_acc = Val_Correct/Num_Val
    val_loss_his.append(np.mean(loss_epoch_val))
    scheduler.step(np.mean(loss_epoch_val))
    print(f'Epoch: {eph} Val Loss: {np.mean(np.array(loss_epoch_val, dtype=np.float32)):10.7f} \
          : Val Acc {val_acc:10.7f}')
    
    if np.mean(loss_epoch_val) <= min(val_loss_his):
        stop = 0
        if eph > 0 :
            print(f'Loss Validation improves from {val_loss_his[eph-1]:.7f} to {val_loss_his[eph]:.7f}')
            torch.save(model.state_dict(), 
                       'Weights\\EvalPad_ResNet18_LSTM_64_30_60_Classes.pt')
            print('Save best weight!')
            
    if np.mean(loss_epoch_val) > min(val_loss_his) :
        stop += 1
        print(f'Loss Validation does not improve from {min(val_loss_his):.7f}')
        print(f'patience ..... {stop} .....')
        if stop == 9:
            print(f'stop training!')
            break
        
    if (np.mean(loss_epoch_val) < min(val_loss_his)) and (np.mean(loss_epoch_val) - min(val_loss_his)) < 0.0001 :
        stop += 1
        print(f'Loss Validation does not improve from {min(val_loss_his):.7f}')
        print(f'patience ..... {stop} .....')
        if stop == 9:
            print(f'stop training!')
            break

# print('Save Complete on SkimPad Resnet18 on the flies 224!')
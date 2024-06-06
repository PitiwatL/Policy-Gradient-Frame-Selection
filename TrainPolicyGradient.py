import numpy as np # linear algebra
import pandas as pd
from numpy.random import choice
from torchvision import transforms

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.models as models

import torch.nn.functional as F
from torchvision.utils import make_grid

from PIL import Image
# import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import train_test_split
import os

import tqdm
from tqdm import tqdm
import time

from Utils.ImageDataset import ImageDataset_Policy_Pad
from Utils.skim_net import SkimNetworkPad, PolicyNetworkPad, EvalNetworkPad
from Utils.policy_gradient import PolicyNetworkPad_PolicyGradient

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

for path in os.listdir(Target_PATH) :
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

train_transform1 = transforms.Compose([
        transforms.Resize((64, 64)),
        # transforms.CenterCrop(224),          ##########################
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

train_transform2 = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.CenterCrop(224),          ##########################
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    
train_dataset = ImageDataset_Policy_Pad(data_paths = X_train,
                                        labels     = y_train,
                                        transform  = (train_transform1, train_transform2), 
                                        target_num_frame = 30, 
                                        mode = 'train' )

val_dataset   = ImageDataset_Policy_Pad(data_paths = X_val, 
                                        labels     = y_val,
                                        transform  = (train_transform1, train_transform2),
                                        target_num_frame = 30, mode = 'test')

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False)

#### Define the based model
SkimNetworkPad = SkimNetworkPad(128)
evalNetwork = EvalNetworkPad(128)

SkimNetworkPad = SkimNetworkPad.to(device)
SkimNetworkPad = torch.nn.DataParallel(SkimNetworkPad)

evalNetwork = evalNetwork.to(device)
evalNetwork = torch.nn.DataParallel(evalNetwork)

SkimNetworkPad.load_state_dict(torch.load('Weights\\SkimPad_ResNet18_LSTM_64_30_60_Classes.pt'))
print("Load Skim weights complete!")

evalNetwork.load_state_dict(torch.load('Weights\\EvalPad_ResNet18_LSTM_64_30_60_Classes.pt'))
print("Load Eval weights complete!")

SkimNetworkPad.module.cf3 = nn.Identity() # pop the last layer out!

model_base = PolicyNetworkPad_PolicyGradient(SkimNetworkPad)
                            
model = model_base.to(device)
model = torch.nn.DataParallel(model)

########################
###### load Weight #####
########################
# model.load_state_dict(torch.load('/project/lt200210-action/OCS_Sampler/weights/Eval_ResNet18_LSTM_Skim_224_5.pt'))

# print(model.module.eval_net.module)
# #### Start Training Loop
# def prep_eval_patch(Full_Frame224, dist):
#     selected_index_set = []
#     selected_index_set_uni = []
#     transform = transforms.Compose([transforms.Resize((224, 224))])
#     for batch_idx in range(dist.shape[0]):
#         dist_1    = sorted(list(dist[batch_idx]), reverse = True) 
#         selected_index_set.append(sorted([list(dist[batch_idx]).index(dist_1[0]), list(dist[batch_idx]).index(dist_1[1]),
#                                           list(dist[batch_idx]).index(dist_1[2]), list(dist[batch_idx]).index(dist_1[3]), 
#                                           list(dist[batch_idx]).index(dist_1[4])]))

#         selected_order_index = sorted(choice([idx for idx in  range(30)], 5 , p = [1/30 for _ in range(30)]))   
#         selected_index_set_uni.append(selected_order_index)                                       

#     selected_frame     = torch.zeros(Full_Frame224.shape[0], 3, 224, 224).to(device)
#     selected_frame_uni = torch.zeros(Full_Frame224.shape[0], 3, 224, 224).to(device)
#     for batch_idx in range(dist.shape[0]):
#         Concat_Frame_list1 = [Full_Frame224[batch_idx, idx, :, :, :] for idx in selected_index_set[batch_idx]]
#         Concat_Frame_list2 = [Full_Frame224[batch_idx, idx, :, :, :] for idx in selected_index_set_uni[batch_idx]]

#         Concat_Frame = torch.cat((Concat_Frame_list1[0],  Concat_Frame_list1[1], Concat_Frame_list1[2], 
#                                   Concat_Frame_list1[3],  Concat_Frame_list1[4]), 1)

#         Concat_Frame_uni = torch.cat((Concat_Frame_list2[0],  Concat_Frame_list2[1], Concat_Frame_list2[2], 
#                                   Concat_Frame_list2[3],  Concat_Frame_list2[4]), 1)

#         selected_frame[batch_idx, :, :, :] += transform(Concat_Frame)
#         selected_frame_uni[batch_idx, :, :, :] += transform(Concat_Frame_uni)


#     return selected_frame, selected_frame_uni

def prep_eval_patch(Full_Frame224, dist):
    # Convert dist to a PyTorch tensor if it is a NumPy array
    if isinstance(dist, np.ndarray):
        dist = torch.tensor(dist, device=device)
    
    transform = transforms.Compose([transforms.Resize((224, 224))])

    # Get the top 5 indices for each batch
    _, selected_index_set = torch.topk(dist, 5, dim=1, largest=True, sorted=True)

    # Uniformly random selection of indices
    selected_order_index   = torch.multinomial(torch.ones(dist.shape[1], device=device), 5, replacement=False)
    selected_index_set_uni = selected_order_index.expand(dist.shape[0], -1)

    # Select the frames based on indices
    Full_Frame224 = Full_Frame224.to(device)

    # Create selected frames
    selected_frames     = Full_Frame224[torch.arange(Full_Frame224.shape[0], device=device).unsqueeze(1), selected_index_set]
    selected_frames_uni = Full_Frame224[torch.arange(Full_Frame224.shape[0], device=device).unsqueeze(1), selected_index_set_uni]

    # Concatenate along the channel dimension and apply transformation
    Concat_Frame     = torch.cat(torch.unbind(selected_frames, dim=1), dim=2)
    Concat_Frame_uni = torch.cat(torch.unbind(selected_frames_uni, dim=1), dim=2)

    selected_frame = transform(Concat_Frame)
    selected_frame_uni = transform(Concat_Frame_uni)

    return selected_frame, selected_frame_uni

PolicyNet = nn.Sequential(nn.ReLU(nn.Linear(64, 64)), 
                                  nn.Linear(64, 30), 
                                  nn.Softmax(dim=-1))
PolicyNet = PolicyNet.to(device)

optimizer = torch.optim.Adam(PolicyNet.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.6, 
                                     patience=3, min_lr= 1 * 1e-6 ,verbose = True)

epochs = 100
val_loss_his = []
train_loss_his = []
stop = 0
criterion = torch.nn.CrossEntropyLoss()
for eph in range(epochs):
    train_preds = []
    train_target = []
    loss_epoch_train=[]
    loss_epoch_val = []
    
    Train_Correct = 0
    Num_Train = 0
    
    # Run the training batches
    PolicyNet.train()
    model.train()
    epoch_loss = 0  
    for b, (X_train1, X_train2, y_train) in tqdm(enumerate(train_loader), total=len(train_loader)):
        
        X_train1, X_train2 = X_train1.to(device), X_train2.to(device)
        y_train = y_train.to(device)
        
        fea_1 = model(X_train1)
        probs = PolicyNet(fea_1)

        dist_ = probs.clone().detach().cpu().numpy()

        selected_skim, uniform_skim = prep_eval_patch(X_train2, dist_)

        evalNetwork.eval()
        with torch.no_grad():
            reward1 =  nn.Softmax(dim=1)(evalNetwork(selected_skim)).detach().cpu().numpy()
            uniform =  nn.Softmax(dim=1)(evalNetwork(uniform_skim)).detach().cpu().numpy()
        
        ground_truth_list = list(np.argmax(uniform, axis=1))
        
        Expectation, reward_list = [], []
        for idx, gr_th in enumerate(ground_truth_list) :
            Expectation.append(uniform[idx][y_train[idx]])
            reward_list.append(reward1[idx][y_train[idx]] - sum(Expectation)/X_train1.shape[0])

        # considering prob as an max element in list , might not be correct, but can get rough approximation
        DiscountedReturns = []
        for t in range(len(reward_list)):
            G = 0.0
            for k, r in enumerate(reward_list[t:]):
                G += 1.0 * k * r
            DiscountedReturns.append(G)
        
        for idx, G in enumerate(DiscountedReturns):
            dist   = torch.distributions.Categorical(probs=probs[idx])  
            action = dist.sample().item()
            Action = torch.tensor(action, dtype=torch.int).to(device)
            log_prob = dist.log_prob(Action)
            # print(log_prob)
            loss = - log_prob * G
            
            epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    epoch_avr = epoch_loss / len(train_loader)
    print(f'epoch: {eph:2}  Train loss: {epoch_avr:10.5f}' )

    torch.save(PolicyNet.state_dict(), 
               'Weights\\PolicyGrad\\Policy_' + str(eph) + "_" + str(epoch_avr) + '.pt')

    # # Run the validation batches
    # Val_Correct = 0
    # Num_Val = 0
    # PolicyNet.eval()
    # evalNetwork.eval()
    # model.eval()
    # with torch.no_grad():
    #     for b, (X_val1, X_val2, y_val) in enumerate(val_loader):
    #         X_val1, X_val2 = X_val1.to(device), X_val2.to(device) 
    #         y_val = y_val.to(device)

    #         fea_1 = model(X_train1)
    #         probs = PolicyNet(fea_1)

    #         dist_ = probs.clone().detach().cpu().numpy()

    #         selected_skim, uniform_skim = prep_eval_patch(X_val2, dist_)

    #         y_val_ = evalNetwork(selected_skim)
    #         y_val_ = torch.argmax(y_val_, 1)

    #         Val_Correct += torch.sum(y_val_ == y_val)
            
    #         Num_Val += y_val.shape[0]
            
    #     val_acc = Val_Correct/Num_Val

    #     print(f'Epoch: {eph} Val Acc {val_acc:10.4f}')
    
#     if np.mean(loss_epoch_val) <= min(val_loss_his):
#         stop = 0
#         if eph > 0 :
#             print(f'Loss Validation improves from {val_loss_his[eph-1]:.7f} to {val_loss_his[eph]:.7f}')
#             torch.save(model.state_dict(), 
#                        '/project/lt200210-action/OCS_Sampler/weights/Policy_224_5_Pad.pt')
#             print('Save best weight!')
            
#     if np.mean(loss_epoch_val) > min(val_loss_his) :
#         stop += 1
#         print(f'Loss Validation does not improve from {min(val_loss_his):.7f}')
#         print(f'patience ..... {stop} .....')
#         if stop == 9:
#             print(f'stop training!')
#             break
        
#     if (np.mean(loss_epoch_val) < min(val_loss_his)) and (np.mean(loss_epoch_val) - min(val_loss_his)) < 0.0001 :
#         stop += 1
#         print(f'Loss Validation does not improve from {min(val_loss_his):.7f}')
#         print(f'patience ..... {stop} .....')
#         if stop == 9:
#             print(f'stop training!')
#             break

# print('Save Complete on Policy on Pad 64!')
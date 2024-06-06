import numpy as np
from numpy.random import choice

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.models as models
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchvision import transforms

from Utils.LSTM import LSTMCells
from Utils.ImageDataset import random_frame_prep

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# def prep_skim_patch(Full_Frame64):
#     transform = transforms.Compose([transforms.Resize((64, 64))])
#     selected_skim = torch.zeros(Full_Frame64.shape[0], 3, 64, 64).to(device)
#     for batch_idx in range(Full_Frame64.shape[0]):
#         Cat_Skim = [Full_Frame64[batch_idx, idx, :, :, :] for idx in range(30)]
#         Cat = torch.cat((Cat_Skim[0],  Cat_Skim[1], Cat_Skim[2], Cat_Skim[3], Cat_Skim[4], Cat_Skim[5],  Cat_Skim[6], Cat_Skim[7], Cat_Skim[8], Cat_Skim[9], 
#                         Cat_Skim[10], Cat_Skim[11], Cat_Skim[12], Cat_Skim[13], Cat_Skim[14], Cat_Skim[15], Cat_Skim[16], Cat_Skim[17], Cat_Skim[18], Cat_Skim[19], 
#                         Cat_Skim[20], Cat_Skim[21], Cat_Skim[22], Cat_Skim[23], Cat_Skim[24], Cat_Skim[25], Cat_Skim[26], Cat_Skim[27], Cat_Skim[28], Cat_Skim[29]), 1)

#         selected_skim[batch_idx, :, :, :] += transform(Cat)

#     return selected_skim

# def prep_eval_patch(Full_Frame224, dist):
#     selected_index_set = []
#     selected_index_set_uni = []
#     transform = transforms.Compose([transforms.Resize((224, 224))])
#     for batch_idx in range(dist.shape[0]):
#         dist_1    = sorted(list(dist[batch_idx]), reverse = True) 
#         selected_index_set.append(sorted([list(dist[batch_idx]).index(dist_1[0]), list(dist[batch_idx]).index(dist_1[1]),
#                                           list(dist[batch_idx]).index(dist_1[2]), list(dist[batch_idx]).index(dist_1[3]), 
#                                           list(dist[batch_idx]).index(dist_1[4])]))

#         selected_order_index = sorted(choice([idx for idx in  range(30)] ,5 , p = [1/30 for _ in range(30)]))   
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

def prep_skim_patch(Full_Frame64):
    transform = transforms.Compose([transforms.Resize((64, 64))])
    
    # Select all frames across batch and indices
    selected_skim = Full_Frame64.permute(0, 2, 3, 4, 1).reshape(Full_Frame64.shape[0], 3, 64, -1)
    
    # Concatenate along the width (dim 3)
    selected_skim = selected_skim.view(Full_Frame64.shape[0], 3, 64, 64 * 30)
    
    # Apply the transform
    selected_skim = transform(selected_skim.to(device))
    
    return selected_skim

def prep_eval_patch(Full_Frame224, dist):
    transform = transforms.Compose([transforms.Resize((224, 224))])

    # Ensure dist is a PyTorch tensor
    if isinstance(dist, np.ndarray):
        dist = torch.tensor(dist)
    
    # Get the top 5 indices for each batch
    _, selected_index_set = torch.topk(dist, 5, dim=1, largest=True, sorted=True)

    # Uniformly random selection of indices
    selected_order_index = torch.multinomial(torch.ones(dist.shape[1]), 5, replacement=False)
    selected_index_set_uni = selected_order_index.expand(dist.shape[0], -1)
    
    # Select the frames based on indices
    Full_Frame224 = Full_Frame224.to(device)
    
    # Create selected frames
    selected_frames = Full_Frame224[torch.arange(Full_Frame224.shape[0]).unsqueeze(1), selected_index_set]
    selected_frames_uni = Full_Frame224[torch.arange(Full_Frame224.shape[0]).unsqueeze(1), selected_index_set_uni]
    
    # Concatenate along the channel dimension
    Concat_Frame = torch.cat(torch.unbind(selected_frames, dim=1), dim=1)
    Concat_Frame_uni = torch.cat(torch.unbind(selected_frames_uni, dim=1), dim=1)
    
    # Apply the transform
    selected_frame = transform(Concat_Frame)
    selected_frame_uni = transform(Concat_Frame_uni)
    
    return selected_frame, selected_frame_uni

class PolicyNetworkPad_PolicyGradient(nn.Module):
    def __init__(self, Skim_network):
        super(PolicyNetworkPad_PolicyGradient, self).__init__()

        self.skim_net = Skim_network.to(device)

    def forward(self, Full_Frame64): # Full_Frame 20 frames
        selected_skim = prep_skim_patch(Full_Frame64)
      
        fea_1 = self.skim_net(selected_skim) # out 64 feed 20 frames # Full_Frame [B, num_frames, 3, 64, 64]

        return fea_1
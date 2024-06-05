import numpy as np

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

class LSTM(nn.Module):
      def __init__(self, input_fea, hidden_unit, return_sequence = False):
        super(LSTM, self).__init__()
        self.hidden_unit = hidden_unit
        self.return_sequence = return_sequence
        self.cell1 = LSTMCells(input_fea = input_fea, hidden_unit = self.hidden_unit)
        self.cell2 = LSTMCells(input_fea = input_fea, hidden_unit = self.hidden_unit)
        
        self.resnet = models.resnet18(weights = None)
        self.resnet.load_state_dict(torch.load(
                                '/project/lt200210-action/OCS_Sampler/Pretrain_weight/ResNet18/resnet18-f37072fd.pth') )
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, 128) 
      
      def forward(self, Seq): # [batch, Num_Cells, Depth]
        ct  = torch.zeros(Seq.shape[0], self.hidden_unit).to(device) 
        ht  = torch.zeros(Seq.shape[0], self.hidden_unit).to(device)
        
        return_seq = torch.zeros(Seq.shape[0], Seq.shape[1], self.hidden_unit).to(device)
        for t in range(Seq.shape[1]): # [B, F, 3, 224, 224]
            ext = self.resnet(Seq[:, t, :, :, :])
            ct, ht   = self.cell1(ext, ct, ht)
            
            return_seq[:, t, :] += ht
        
        ct_ = torch.zeros(Seq.shape[0], self.hidden_unit).to(device) 
        ht_ = torch.zeros(Seq.shape[0], self.hidden_unit).to(device)
        for t in range(return_seq.shape[1]):
            ct_, ht_ = self.cell2(return_seq[:, t, :], ct_, ht_)
          
        if self.return_sequence == False :    
            
            return ht_

class SkimNetwork(nn.Module):
    def __init__(self, hidden_unit):
        super(SkimNetwork, self).__init__()

        self.hidden_dim = hidden_unit
        # initialize the attention blocks defined above
        self.LSTM1 = LSTM(128, 128, return_sequence = False)
        
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(128)
        self.batchnorm4 = nn.BatchNorm1d(64)

        self.cf1 = nn.Linear(hidden_unit, 128)
        self.cf2 = nn.Linear(128, 64)
        self.cf3 = nn.Linear(64, 60)

        # self.dropout = nn.Dropout(0.2)
        self.relu  = nn.ReLU()


        # self.batchnorm1 = nn.BatchNorm1d(10, 256)
       
    def forward(self, Seq):
        x  = self.LSTM1(Seq)
        x  = self.batchnorm2(x)
        # x  = self.dropout(x)
        
        x  = self.relu(self.batchnorm3(self.cf1(x)))
        # x  = self.dropout(x)
        
        x  = self.relu(self.cf2(x))
        out  = self.cf3(x)
        
        return out

class EvalNetwork(nn.Module):
    def __init__(self, hidden_unit):
        super(EvalNetwork, self).__init__()

        self.hidden_dim = hidden_unit
        self.cf1 = nn.Linear(hidden_unit, 128)
        self.cf2 = nn.Linear(128, 64)
        self.cf3 = nn.Linear(64, 60)

        # self.dropout = nn.Dropout(0.2)
        self.relu  = nn.ReLU()

        # initialize the attention blocks defined above
        self.LSTM1 = LSTM(128, 128, return_sequence = False)

        # self.batchnorm1 = nn.BatchNorm1d(10, 256)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(128)
        self.batchnorm4 = nn.BatchNorm1d(64)
       
    def forward(self, Seq):
        x  = self.LSTM1(Seq)
        x  = self.batchnorm2(x)
        # x  = self.dropout(x)
        
        x  = self.relu(self.batchnorm3(self.cf1(x)))
        # x  = self.dropout(x)
        
        x  = self.relu(self.cf2(x))
        out  = self.cf3(x)
        
        return out

class SkimNetworkPad(nn.Module):
    def __init__(self, hidden_unit):
        super(SkimNetworkPad, self).__init__()
        self.resnet = models.resnet18(weights = None)
        self.resnet.load_state_dict(torch.load(
                                '/project/lt200210-action/OCS_Sampler/Pretrain_weight/ResNet18/resnet18-f37072fd.pth') )
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, 128) 

        self.hidden_dim = hidden_unit
        self.cf1 = nn.Linear(hidden_unit, 128)
        self.cf2 = nn.Linear(128, 64)
        self.cf3 = nn.Linear(64, 60)

        # self.dropout = nn.Dropout(0.2)
        self.relu  = nn.ReLU()

        # self.batchnorm1 = nn.BatchNorm1d(10, 256)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(128)
       
    def forward(self, img_pad):
        a = self.resnet(img_pad)
        b = self.cf1(a)
        c = self.relu(b).clone()
        d = self.cf2(c)
        e = self.relu(d).clone()
        
        out  = self.cf3(e)
        
        return out

class EvalNetworkPad(nn.Module):
    def __init__(self, hidden_unit):
        super(EvalNetworkPad, self).__init__()
        self.resnet = models.resnet18(weights = None)
        self.resnet.load_state_dict(torch.load(
                                '/project/lt200210-action/OCS_Sampler/Pretrain_weight/ResNet18/resnet18-f37072fd.pth') )
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, 128) 

        self.hidden_dim = hidden_unit
        self.cf1 = nn.Linear(hidden_unit, 128)
        self.cf2 = nn.Linear(128, 64)
        self.cf3 = nn.Linear(64, 60)

        # self.dropout = nn.Dropout(0.2)
        self.relu  = nn.ReLU()

        # self.batchnorm1 = nn.BatchNorm1d(10, 256)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(128)
       
    def forward(self, img_pad):
        x  = self.batchnorm1(self.resnet(img_pad))
        # x  = self.dropout(x)
        x  = self.relu(self.batchnorm2(self.cf1(x)))
        # x  = self.dropout(x)
        x  = self.relu(self.cf2(x))
        out  = self.cf3(x)
        
        return out

class PolicyNetwork(nn.Module):
    def __init__(self, Skim_network, Eval_network):
        super(PolicyNetwork, self).__init__()

        self.skim_net = Skim_network.to(device)
        self.eval_net = Eval_network.to(device)

        # # only return gradient but not update weights
        # for param in self.eval_net.parameters():
        #     param.detach_()
            
        self.PolicyNet = nn.Sequential(nn.ReLU(nn.Linear(64, 64)), 
                                       nn.Linear(64, 20))

        self.relu  = nn.ReLU()

    def forward(self, Full_Frame64, Full_Frame224, mode): # Full_Frame 20 frames
    
        fea_1 = self.skim_net(Full_Frame64) # out 64 feed 20 frames # Full_Frame [B, num_frames, 3, 64, 64]
        dist = torch.nn.Softmax(dim=1)(self.PolicyNet(fea_1)).detach().cpu().numpy() # distribution
        
        count = 0
        selected_index_set = []

        if mode == 'Train':
            selected_frame = torch.zeros(Full_Frame224.shape[0], 5, 3, 224, 224).to(device)

            for batch_idx in range(dist.shape[0]):
                dist_1 = dist[batch_idx]
                min_index = np.argmin(dist[batch_idx])
                
                dist_1[min_index] = 0
                complement = 1 - sum(dist_1)
                dist_1[min_index] = complement # make the sum of prob equals to 1

                selected_index = sorted(np.random.choice([idx for idx in range(20)], 5,  p = dist_1 ))
                selected_index_set.append(selected_index)

            for batch_idx in range(dist.shape[0]):
                for idx, idx_frame in enumerate(sorted(selected_index_set[batch_idx])) :
                    selected_frame[batch_idx, idx, :, :, :] += Full_Frame224[batch_idx, idx_frame, :, :, :]

            out = self.eval_net(selected_frame)
        
        if mode == 'Test' :
            selected_frame = torch.zeros(Full_Frame224.shape[0], 5, 3, 224, 224).to(device)

            for batch_idx in range(dist.shape[0]):
                dist_1    = sorted(list(dist[batch_idx]), reverse = True) 
                selected_index_set.append(sorted([list(dist[batch_idx]).index(dist_1[0]), list(dist[batch_idx]).index(dist_1[1]),
                                        list(dist[batch_idx]).index(dist_1[2]), list(dist[batch_idx]).index(dist_1[3]), 
                                        list(dist[batch_idx]).index(dist_1[4])]))

            for batch_idx in range(dist.shape[0]):
                for idx, idx_frame in enumerate(selected_index_set[batch_idx]) :
                    selected_frame[batch_idx, idx, :, :, :] += Full_Frame224[batch_idx, idx_frame, :, :, :]

            out = self.eval_net(selected_frame) # out [B, 10]

        return out

class PolicyNetworkPad(nn.Module):
    def __init__(self, Skim_network, Eval_network_pad):
        super(PolicyNetworkPad, self).__init__()

        self.skim_net = Skim_network.to(device)
        self.eval_net = Eval_network_pad.to(device)

        # # only return gradient but not update weights
        # for param in self.eval_net.parameters():
        #     param.detach_()
            
        self.PolicyNet = nn.Sequential(nn.ReLU(nn.Linear(64, 64)), 
                                       nn.Linear(64, 30))

        self.relu  = nn.ReLU()

    def forward(self, Full_Frame64, Full_Frame224): # Full_Frame 20 frames
        transform = transforms.Compose([transforms.Resize((64, 64))])
        selected_skim = torch.zeros(Full_Frame64.shape[0], 3, 64, 64).to(device)
        for batch_idx in range(Full_Frame64.shape[0]):
            Cat_Skim = [Full_Frame64[batch_idx, idx, :, :, :] for idx in range(30)]
            Cat = torch.cat((Cat_Skim[0],  Cat_Skim[1], Cat_Skim[2], Cat_Skim[3], Cat_Skim[4], Cat_Skim[5],  Cat_Skim[6], Cat_Skim[7], Cat_Skim[8], Cat_Skim[9], 
                            Cat_Skim[10], Cat_Skim[11], Cat_Skim[12], Cat_Skim[13], Cat_Skim[14], Cat_Skim[15], Cat_Skim[16], Cat_Skim[17], Cat_Skim[18], Cat_Skim[19], 
                            Cat_Skim[20], Cat_Skim[21], Cat_Skim[22], Cat_Skim[23], Cat_Skim[24], Cat_Skim[25], Cat_Skim[26], Cat_Skim[27], Cat_Skim[28], Cat_Skim[29]), 1)

            selected_skim[batch_idx, :, :, :] += transform(Cat)

        fea_1 = self.skim_net(selected_skim) # out 64 feed 20 frames # Full_Frame [B, num_frames, 3, 64, 64]
        dist = torch.nn.Softmax(dim=1)(self.PolicyNet(fea_1)).detach().cpu().numpy() # distribution
        
        count = 0
        selected_index_set = []
        
        transform = transforms.Compose([transforms.Resize((224, 224))])

        selected_frame = torch.zeros(Full_Frame224.shape[0], 3, 224, 224).to(device)
        for batch_idx in range(dist.shape[0]):
            dist_1    = sorted(list(dist[batch_idx]), reverse = True) 
            selected_index_set.append(sorted([list(dist[batch_idx]).index(dist_1[0]), list(dist[batch_idx]).index(dist_1[1]),
                                              list(dist[batch_idx]).index(dist_1[2]), list(dist[batch_idx]).index(dist_1[3]), 
                                              list(dist[batch_idx]).index(dist_1[4])]))

        for batch_idx in range(dist.shape[0]):
            Concat_Frame_list = [Full_Frame224[batch_idx, idx, :, :, :] for idx in selected_index_set[batch_idx]]
            Concat_Frame = torch.cat((Concat_Frame_list[0],  Concat_Frame_list[1], Concat_Frame_list[2], 
                                      Concat_Frame_list[3],  Concat_Frame_list[4]), 1)

            selected_frame[batch_idx, :, :, :] += transform(Concat_Frame)


        # print(selected_frame.shape)
        out = self.eval_net(selected_frame) # out [B, 10]

        return out
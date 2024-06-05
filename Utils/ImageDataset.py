import torch
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms
from numpy.random import choice
import numpy as np
import os

def fixed_frame(vid_path, train_transform, size = 224 ,target_num_frame = 10) :

    selected = torch.zeros(target_num_frame, 3, size, size)
    cap = cv2.VideoCapture(vid_path)
    num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0
    for idx_frame in range(num_frame) :
        ret, frame = cap.read()
        for_every = num_frame // target_num_frame
        if idx_frame%for_every == 0 :
            count += 1
            selected += train_transform(Image.fromarray(frame))
            if count == target_num_frame : break
                
    return selected

def random_frame(vid_path, train_transform, size = 224, target_num_frame = 10) :
    selected = torch.zeros(target_num_frame, 3, size, size)
    cap = cv2.VideoCapture(vid_path)
    num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    count = 0
    selected_index = sorted(choice([idx+1 for idx in range(num_frame)], target_num_frame, 
                                   p= [1/num_frame for _ in range(num_frame)]))
    
    for idx_frame in range(num_frame) :
        ret, frame = cap.read()
        
        if idx_frame in selected_index:
            count += 1
            img = train_transform(Image.fromarray(frame))
            # print(img.shape)
            selected += img
            
            if count == target_num_frame : 
                break
                
    return selected

def fixed_frame_prep(vid_path, train_transform, size = 224 ,target_num_frame = 10) :

    selected = torch.zeros(target_num_frame, 3, size, size)

    num_frame = len(os.listdir(vid_path))
    count = 0
    for idx_frame, frame_path in enumerate(sorted(os.listdir(vid_path))) :
        for_every = num_frame // target_num_frame
        if idx_frame%for_every == 0 :
            count += 1
            # print(os.listdir(vid_path))
            frame = cv2.imread(vid_path + "/" + frame_path )
            selected += train_transform(Image.fromarray(frame))
            if count == target_num_frame : break
                
    return selected

def random_frame_prep(vid_path, train_transform, size = 224, target_num_frame = 10) :
    selected = torch.zeros(target_num_frame, 3, size, size)

    num_frame = len(os.listdir(vid_path))
    
    count = 0
    selected_order = sorted(choice([frame_order.split('.')[0] for frame_order in sorted(os.listdir(vid_path)) ], 
                                    target_num_frame, 
                                   p = [1/num_frame for _ in range(num_frame)]))
        
    for idx_frame in selected_order:
        count += 1
        # print(os.listdir(vid_path))
        frame = cv2.imread(vid_path + "/" + idx_frame + '.jpg')

        selected += train_transform(Image.fromarray(frame))
        
        if count == target_num_frame : 
            break
                
    return selected

def fixed_frame_prep_pad(vid_path, train_transform, size = 224 ,target_num_frame = 10) :
    
    Frame_List = []
    # selected = np.zeros(target_num_frame, 3, size, size)

    num_frame = len(os.listdir(vid_path))
    count = 0
    for idx_frame, frame_path in enumerate(sorted(os.listdir(vid_path))) :
        for_every = num_frame // target_num_frame
        if idx_frame%for_every == 0 :
            count += 1
            
            frame = cv2.imread(vid_path + "/" + frame_path )
            Frame_List.append(frame)
           
            if count == target_num_frame : break

    # t = [frame_ for frame_ in Frame_List]
    Cat_Frame = np.concatenate(Frame_List , axis=1) 
                
    return train_transform(Image.fromarray(Cat_Frame))

def random_frame_prep_pad(vid_path, train_transform, size = 224, target_num_frame = 10) :
    Frame_List = []
    # selected = np.zeros(target_num_frame, 3, size, size)
    num_frame = len(os.listdir(vid_path))
    
    count = 0
    selected_order = sorted(choice([frame_order.split('.')[0] for frame_order in sorted(os.listdir(vid_path)) ], 
                                    target_num_frame, 
                                   p = [1/num_frame for _ in range(num_frame)]))
        
    for idx_frame in selected_order:
        count += 1
        # print(os.listdir(vid_path))
        frame = cv2.imread(vid_path + "/" + idx_frame + '.jpg')
        Frame_List.append(frame)
        
        if count == target_num_frame : break

    # t = (frame_ for frame_ in Frame_List)
    Cat_Frame = np.concatenate(Frame_List , axis=1) 
    
    return train_transform(Image.fromarray(Cat_Frame))



class ImageDataset1(Dataset):
    def __init__(self, data_paths, labels, size = 224, target_num_frame = 10, 
                 transform=None, mode='train'):
        self.data   = data_paths
        self.labels = labels
        self.size   = size
        self.target_num_frame = target_num_frame
        self.transform = transform
        self.mode   = mode
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        vid_path = self.data[idx]
        labels = torch.tensor(self.labels[idx])
        
        if self.mode == 'train' :
            selected_frames = random_frame(vid_path,  self.transform,  
                                            size = self.size, target_num_frame = self.target_num_frame)

            return selected_frames, labels

        if self.mode == 'test' :
            selected_frames = fixed_frame(vid_path,  self.transform,  
                                            size = self.size, target_num_frame = self.target_num_frame)

            return selected_frames, labels

class ImageDataset2(Dataset):
    def __init__(self, data_paths, labels, size = 224, target_num_frame = 10, 
                 transform=None, mode='train'):
        self.data   = data_paths
        self.labels = labels
        self.size   = size
        self.target_num_frame = target_num_frame

        self.transform = transform

        self.mode   = mode
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        vid_path = self.data[idx]
        labels = torch.tensor(self.labels[idx])
        
        if self.mode == 'train' :
            selected_frames = random_frame_prep(vid_path,  self.transform,  
                                            size = self.size, target_num_frame = self.target_num_frame)

            return selected_frames, labels

        if self.mode == 'test' :
            selected_frames = fixed_frame_prep(vid_path,  self.transform,  
                                            size = self.size, target_num_frame = self.target_num_frame)

            return selected_frames, labels

class ImageDatasetPad(Dataset):
    def __init__(self, data_paths, labels, size = 224, target_num_frame = 10, 
                 transform=None, mode='train'):
        self.data   = data_paths
        self.labels = labels
        self.size   = size
        self.target_num_frame = target_num_frame

        self.transform = transform

        self.mode   = mode
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        vid_path = self.data[idx]
        labels = torch.tensor(self.labels[idx])
        
        if self.mode == 'train' :
            selected_frames = random_frame_prep_pad(vid_path,  self.transform,  
                                            size = self.size, target_num_frame = self.target_num_frame)

            return selected_frames, labels

        if self.mode == 'test' :
            selected_frames = fixed_frame_prep_pad(vid_path,  self.transform,  
                                            size = self.size, target_num_frame = self.target_num_frame)

            return selected_frames, labels

class ImageDataset_Policy(Dataset):
    def __init__(self, data_paths, labels, size = 224, 
                 target_num_frame = 20, 
                 transform=None, mode='train'):
        self.data   = data_paths
        self.labels = labels
        self.size   = size
        self.target_num_frame = target_num_frame

        self.transform1 = transform[0]
        self.transform2 = transform[1]

        self.mode   = mode
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        vid_path = self.data[idx]
        labels = torch.tensor(self.labels[idx])
        
        if self.mode == 'train' :
            selected_frames_skim = random_frame_prep(vid_path,  self.transform1,  
                                            size = 64, target_num_frame = self.target_num_frame)

            selected_frames_eval = random_frame_prep(vid_path,  self.transform2,  
                                            size = 224, target_num_frame = self.target_num_frame)

            return selected_frames_skim, selected_frames_eval, labels

        if self.mode == 'test' :
            selected_frames_skim = fixed_frame_prep(vid_path,  self.transform1,  
                                            size = 64, target_num_frame = self.target_num_frame)

            selected_frames_eval = fixed_frame_prep(vid_path,  self.transform2,  
                                            size = 224, target_num_frame = self.target_num_frame)
                                            
            return selected_frames_skim, selected_frames_eval, labels

class ImageDataset_Policy_Pad(Dataset):
    def __init__(self, data_paths, labels, size = 224, 
                 target_num_frame = 20, 
                 transform=None, mode='train'):
        self.data   = data_paths
        self.labels = labels
        self.size   = size
        self.target_num_frame = target_num_frame

        self.transform1 = transform[0]
        self.transform2 = transform[1]

        self.mode   = mode
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        vid_path = self.data[idx]
        labels = torch.tensor(self.labels[idx])
        
        if self.mode == 'train' :
            selected_frames_skim = random_frame_prep(vid_path,  self.transform1,  
                                            size = 64, target_num_frame = self.target_num_frame)

            selected_frames_eval = random_frame_prep(vid_path,  self.transform2,  
                                            size = 224, target_num_frame = self.target_num_frame)

            return selected_frames_skim, selected_frames_eval, labels

        if self.mode == 'test' :
            selected_frames_skim = fixed_frame_prep(vid_path,  self.transform1,  
                                            size = 64, target_num_frame = self.target_num_frame)

            selected_frames_eval = fixed_frame_prep(vid_path,  self.transform2,  
                                            size = 224, target_num_frame = self.target_num_frame)
                                            
            return selected_frames_skim, selected_frames_eval, labels
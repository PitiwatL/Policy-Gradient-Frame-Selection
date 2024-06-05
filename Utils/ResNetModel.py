import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

Resnet50 = models.resnet50()

class ResNet50(nn.Module):
    def __init__(self, num_classes, dropout=None):
        super(ResNet50, self).__init__()
        Resnet50.load_state_dict(torch.load('/project/lt200048-video/Plueangw/RoiClassification/Pretrain_weight/ResNet50/resnet50-0676ba61.pth'))

        self.pretrain = Resnet50
        
        if dropout is not None:
            self.dpt = nn.Dropout(dropout)
        
        self.fc1 = nn.Linear(in_features = 32000, 
                             out_features= 512, bias=True)
        
        self.fc2 = nn.Linear(in_features = 512, 
                             out_features= num_classes, bias=True)
        
                
    def forward(self, x):
        feature = self.pretrain(x)
        flatten = torch.flatten(feature, start_dim=0, end_dim=- 1)
        out1 = self.fc1(flatten)
        out2 = self.fc2(out1)
        
        return out2
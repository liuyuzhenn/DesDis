import torch.nn as nn
import torch.nn.functional as F


    
class DesDis(nn.Module):
    def __init__(self, dim=32, dim_desc=128, drop_rate=0.1):
        super().__init__()
        self.drop_rate = drop_rate
        norm_layer = nn.BatchNorm2d
        activation = nn.ReLU()
        
        self.layer1 = nn.Sequential(
            nn.InstanceNorm2d(1,affine=False),
            nn.Conv2d(1,dim,kernel_size=3,stride=2,padding=1,bias=False),
            norm_layer(dim,affine=False),
            activation,
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(dim,dim*2,kernel_size=3,stride=2,padding=1,bias=False),
            norm_layer(dim*2,affine=False),
            activation,
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(dim*2,dim*4,kernel_size=3,stride=2,padding=1,bias=False),
            norm_layer(dim*4,affine=False),
            activation,
            nn.Conv2d(dim*4,dim*4,kernel_size=3,stride=1,padding=1,bias=False),
            norm_layer(dim*4,affine=False),
            activation,
        )
        self.desc=nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Conv2d(dim*4,dim_desc,kernel_size=4),
            norm_layer(dim_desc,affine=False),
        )

    def forward(self,x, mode='eval'):
        for layer in [self.layer1,self.layer2,self.layer3]:
            x = layer(x)
        x = self.desc(x)
        
        desc_raw = x.squeeze()
        desc = F.normalize(desc_raw,p=2,dim=1)
        return desc

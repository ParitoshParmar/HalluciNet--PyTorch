#Author: Paritosh Parmar

import torch
import torch.nn as nn
import numpy as np
import random
from opts import randomseed

torch.manual_seed(randomseed); torch.cuda.manual_seed_all(randomseed); random.seed(randomseed); np.random.seed(randomseed)

class Cls_branch(nn.Module): # for ResNet-50
    def __init__(self):
        super(Cls_branch, self).__init__()
        self.fc = nn.Linear(2048, 101) # change according to the dataset

    def forward(self, x):
        x = self.fc(x)
        return x
        
        
class Sidetask_branch_resnext(nn.Module): # for ResNet-50
    def __init__(self):
        super(Sidetask_branch_resnext, self).__init__()
        self.fc = nn.Linear(2048, 2048)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc(x))
        return x

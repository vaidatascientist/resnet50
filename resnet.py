import torch
from torch import nn

class ResNetBlock(nn.Module):
    def __init__(self, in_features, out_features, stride, islayer2, is_first_block):
        super().__init__()
        
        if not islayer2 and is_first_block:
            in_features = in_features * 2
            
        if not is_first_block:
            in_features = in_features * 4
        
        self.expansion = 4
        
        self.conv1 = nn.Conv2d(in_channels=in_features, 
                               out_channels=out_features, 
                               stride=stride,
                               kernel_size=1,
                               padding=0)
        self.bn1 = nn.BatchNorm2d(out_features)
        
        self.conv2 = nn.Conv2d(in_channels=out_features, 
                               out_channels=out_features, 
                               stride=1,
                               kernel_size=3,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(out_features)
        
        self.conv3 = nn.Conv2d(in_channels=out_features, 
                               out_channels=out_features*self.expansion, 
                               stride=1,
                               kernel_size=1,
                               padding=0)
        self.bn3 = nn.BatchNorm2d(out_features*self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.identity_downsample = None
        if in_features != out_features*self.expansion:
            self.identity_downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_features, 
                          out_channels=out_features*self.expansion, 
                          kernel_size=1, 
                          stride=stride, 
                          padding=0),
                nn.BatchNorm2d(out_features*self.expansion)
            )
        
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        out += identity
        out = self.relu(out)
        
        return out


model = nn.Sequential()
model.add_module('Conv1', nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3))
model.add_module('BatchNorm 1', nn.BatchNorm2d(num_features=64))
model.add_module('ReLU 1', nn.ReLU(inplace=True))
model.add_module('MaxPool1', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

previous_size = 64
is_first_block = True  # Variable to track the first block of each feature size
block_index = 0  # Initialize a block index counter

for idx, feature_size in enumerate([64]*3 + [128]*4 + [256]*6 + [512]*3):
    if previous_size != feature_size:
        is_first_block = True  # Reset for the new feature size group
        block_index = 1
    else:
        is_first_block = False  # Not the first block anymore
        block_index += 1
    previous_size = feature_size
    
    if idx == 0:
        is_first_block = True
    # Ensure all blocks in conv2_x layer (feature_size 64) have stride 1
    # Apply stride 2 only for the first block of new feature size groups, except for conv2_x
    if feature_size == 64:
        stride = 1
    else:
        stride = 2 if is_first_block else 1
    
    islayer2 = False if feature_size != 64 else True
    model.add_module(f'{feature_size}_block_{block_index}',  # Added idx to ensure unique names
                     ResNetBlock(in_features=previous_size,
                                 out_features=feature_size,
                                 stride=stride,
                                 islayer2=islayer2,
                                 is_first_block=is_first_block))
    

model.add_module('AvgPoolLast', nn.AdaptiveAvgPool2d((1)))
model.add_module('Flatten', nn.Flatten())
model.add_module('Fully connected Linear', nn.Linear(2048, 10))  # Adjust based on your class_names variable

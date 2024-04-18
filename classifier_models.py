import attention
import torch
from torch import nn

class CNN_Classifier(nn.Sequential):

    def __init__(self):
        return super().__init__(                        # (B, Channels=1, H, W)
            nn.Conv2d(1, 32, kernel_size=3, padding=1), # (B, 32, H, W)
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.MaxPool2d(kernel_size=2),                # (B, 32, H / 2, W / 2)
            nn.Flatten(),                               # (B, 32 * H/2 * W/2) = 8 * H * W = 8 * 784 = 6272
            nn.Linear(6272, 128),                       # (B, 128)
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 10),                         # (B, 10)
            nn.LogSoftmax(dim=1))
    
class UNET_Classifier(nn.Sequential):

    def __init__(self):
        return super().__init__(                        # (B, Channels=1, H, W)
            nn.Conv2d(1, 32, kernel_size=3, padding=1), # (B, 32, H, W)
            nn.ReLU(),
            attention.UNETAttentionBlock(4, 8),
            nn.MaxPool2d(kernel_size=2),                # (B, 32, H / 2, W / 2)
            nn.Flatten(),                               # (B, 32 * H/2 * W/2) = 8 * H * W = 8 * 784 = 6272
            nn.Linear(6272, 128),                       # (B, 128)
            nn.ReLU(),
            nn.Linear(128, 10),                         # (B, 10)
            nn.LogSoftmax(dim=1))
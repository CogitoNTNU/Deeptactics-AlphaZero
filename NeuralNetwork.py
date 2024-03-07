import torch.nn as nn


class ResNet(nn.module):
    def __init__(self):
        super().__init__()
        self.hidden_dimention = 256
        self.input_dimention = 3

        self.initial = nn.Sequential(
            nn.Conv2d(
                self.input_dimention, self.hidden_dimention, kernel_size=3, stride=1
            ),  # Convolution matrix
            nn.BatchNorm2d(self.hidden_dimention),  # Batch normalization
            nn.ReLU() # Activation function
        )

        # [1, 9, 2, 3..]
        self.block = nn.Sequential(
            nn.Conv2d(
                self.hidden_dimention, self.hidden_dimention, kernel_size=3, stride=1
            ),  # Convolution matrix
            nn.BatchNorm2d(self.hidden_dimention),  # Batch normalization
            nn.ReLU(), # Activation function
            nn.Conv2d(
                self.hidden_dimention, self.hidden_dimention, kernel_size=3, stride=1
            ),  # Convolution matrix
            nn.BatchNorm2d(self.hidden_dimention),  # Batch normalization
        )
        



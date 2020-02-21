import torch
import torch.nn as nn

class CNN(nn.Module):

    def __init__(self, n_actions, height=105, width=80, n_channels=4):
        super(CNN, self).__init__()
        channel_sizes = (n_channels, 16, 32)
        kernel_sizes = (8, 4)
        strides = (4, 2)
        
        layers = []
        for i in range(len(channel_sizes)-1):
            layers.append(nn.Conv2d(channel_sizes[i], channel_sizes[i+1], kernel_size=kernel_sizes[i], stride=strides[i]))
            #layers.append(nn.BatchNorm2d(channel_sizes[i+1]))
            layers.append(nn.ReLU())
        
        layers.append(nn.Flatten())
        output_h, output_w = self.conv2d_output_size((height, width), kernel_sizes, strides)
        linear_input_size = output_h * output_w * channel_sizes[-1]
        layers.append(nn.Linear(linear_input_size, 256))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(256, n_actions))
        
        self.main = nn.Sequential(*layers)

    def forward(self, state):
        state = state.float() / 255.0
        return self.main(state)
    
    def conv2d_output_size(self, input_size, kernel_sizes, strides):
        curr_h, curr_w = input_size
        for kernel_size, stride in zip(kernel_sizes, strides):
            curr_h = (curr_h - (kernel_size - 1) - 1) // stride  + 1
            curr_w = (curr_w - (kernel_size - 1) - 1) // stride  + 1
        return curr_h, curr_w

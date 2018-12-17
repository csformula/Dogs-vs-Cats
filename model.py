# -*- coding: utf-8 -*-

import torch.nn as nn

class Res(nn.Module):
    '''
    Transfer learning using Resnet, 
    freeze all network except final layer. 
    '''
    
    def __init__(self, resnet, num_classes):
        '''
        Args:
            resnet(callable, models.ResNet): a pretrained Resnet.
            num_classes(int): num of classes to be predicted.
        '''
        super(Res, self).__init__()
        self.num_classes = num_classes        
        self.model = resnet
        for params in self.model.parameters():
            params.requires_grad = False
        num_in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_in_features, self.num_classes)
        self.params = self.model.fc.parameters()
        
    def forward(self, x):
        return self.model(x)
        
        
class Darknet(nn.Module):
    ''' 
    Defining a net similar to Darknet19,
    training from scratch
    '''
    
    def __init__(self, num_classes):
        '''
        Args:
            num_classes(int): num of classes to be classified.
        '''
        
        super(Darknet, self).__init__()
        self.num_classes = num_classes
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.conv_list = nn.ModuleList([
            ConvBlock(3, 3, 32),
            ConvBlock(3, 32, 64),
            ConvBlock(3, 64, 128),
            ConvBlock(1, 128, 64),
            ConvBlock(3, 64, 128),
            ConvBlock(3, 128, 256),
            ConvBlock(1, 256, 128),
            ConvBlock(3, 128, 256),
            ConvBlock(3, 256, 512),
            ConvBlock(1, 512, 256),
            ConvBlock(3, 256, 512),
            ConvBlock(1, 512, 256),
            ConvBlock(3, 256, 512),
            ConvBlock(3, 512, 1024),
            ConvBlock(1, 1024, 512),
            ConvBlock(3, 512, 1024),
            ConvBlock(1, 1024, 512),
            ConvBlock(3, 512, 1024),
            ConvBlock(1, 1024, self.num_classes)
        ])
    
    def forward(self, x):
        x = self.conv_list[0](x)
        x = self.maxpool(x)
        x = self.conv_list[1](x)
        x = self.maxpool(x)
        x = self.conv_list[2](x)
        x = self.conv_list[3](x)
        x = self.conv_list[4](x)
        x = self.maxpool(x)
        x = self.conv_list[5](x)
        x = self.conv_list[6](x)
        x = self.conv_list[7](x)
        x = self.maxpool(x)
        x = self.conv_list[8](x)
        x = self.conv_list[9](x)
        x = self.conv_list[10](x)
        x = self.conv_list[11](x)
        x = self.conv_list[12](x)
        x = self.maxpool(x)
        x = self.conv_list[13](x)
        x = self.conv_list[14](x)
        x = self.conv_list[15](x)
        x = self.conv_list[16](x)
        x = self.conv_list[17](x)
        x = self.conv_list[18](x)
        x = self.avgpool(x)
        
        return x.squeeze()

class ConvBlock(nn.Module):
    ''' Defining a module to implement conv, bn, leakyrelu at the same time. '''
    
    def __init__(self, size, infilters, outfilters, 
                 stride=1, pad=True, batch_normalize=True):
        ''' 
        Args:
            size(int, tuple): kernel size.
            infilters(int): num of input channels.
            outfilters(int): num of output channels.
            stride(int, optional): stride.
            batch_normalize(bool, optional): batch norm before activation or not.         
            pad(bool, optional): padding or not.
        '''
        super(ConvBlock, self).__init__()
        if pad==True:
            padding = size//2
        else:
            padding = 0
        if batch_normalize:
            bias=False
        else:
            bias=True
        self.conv = nn.Conv2d(infilters, outfilters, size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(outfilters)
        self.leaky = nn.LeakyReLU(inplace=True)
        
        self.block = nn.Sequential(self.conv, 
                                   self.bn, 
                                   self.leaky)
        
    def forward(self, x):
        return self.block(x)
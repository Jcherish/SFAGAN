import argparse
from metrics import *
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from torchvision.utils import save_image
from skimage.metrics import structural_similarity

ndf=32
ngf=32
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.dis=nn.Sequential(
            #输入图像256*256
            nn.Conv2d(2,ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            ## state size ndf*128*128
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            ##state size (ndf*2)*64*64
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride= 2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            ##state size (ndf*4)*32*32
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*8,ndf*16,4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(ndf*16),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(ndf*16,ndf*32,4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(ndf*32),
            nn.LeakyReLU(0.2,inplace=True),
            ##state size (ndf*8)*16*16
            nn.Conv2d(ndf * 32, 1, kernel_size=4, stride=2,padding= 0, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        x = self.dis(x)
        return x.view(-1, 1).squeeze(1)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(

            ##input is z_dimension
            nn.ConvTranspose2d(256, ngf * 32, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 32),
            nn.ReLU(True),

            ##state size (ngf*8)*4*4
            nn.ConvTranspose2d(ngf * 32, ngf * 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),

            ##state size (ngf*4)*8*8
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            ##state size (ngf*2)*16*16
            nn.ConvTranspose2d(ngf * 8, ngf* 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*4,ngf*2,4,2,1,bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*2,ngf,4,2,1,bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            ##state size ngf*256*256
            nn.ConvTranspose2d(ngf, 2, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.gen(x)
        return x

"""Construct the computational graph of model for training. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
from buildingblocks import ResTT, TT, TTBN
import torch.nn.functional as F


# class FullNetTN(nn.Module):
#     def __init__(self, k1=5, c1=40, k2=5, c2=40, d1=96, in_bond=2):
#         super(FullNetTN, self).__init__()
#         self.TT1 = TT(kernel_size=k1, in_bond=in_bond, hidden_bond=2, channels=c1, std=0.01, name='TT1')
#         self.bn1 = nn.BatchNorm2d(c1)
#         self.TT2 = TT(kernel_size=k2, in_bond=c1, hidden_bond=2, channels=c2, std=0.01, name='TT2')
#         self.bn2 = nn.BatchNorm2d(c2)
#         self.fc1 = nn.Linear(4 * 4 * c2, d1)
#         self.fc2 = nn.Linear(d1, 10)
#
#     def forward(self, x):
#         x = self.TT1(x)
#         x = F.relu(self.bn1(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = self.TT2(x)
#         x = F.relu(self.bn2(x))
#         x = F.max_pool2d(x, 2, 2)
#         shape = list(x.size())
#         x = x.view(-1, shape[1] * shape[2] * shape[-1])
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         # x = F.log_softmax(x, dim=1)
#         # shape = list(x.size())
#         # x = x.view(-1, shape[1])
#         return x

class FN_net(nn.Module):
    def __init__(self, d=1, in_x=3, in_y=5):
        super(FN_net, self).__init__()
        self.fc = nn.Linear(in_x * in_y, d)
        self.inx = in_x
        self.iny = in_y

    def forward(self, x):
        x = x.view(-1, self.inx * self.iny)
        # x = F.relu(self.fc1(x))
        x = self.fc(x)
        # x = F.log_softmax(x, dim=1)
        # shape = list(x.size())
        # x = x.view(-1, shape[1])
        return x

class Mtilinear_Net(nn.Module):
    def __init__(self, lengthX=14, lengthY=14,  in_bond=2, hidden_bond=2,out_dim = 10):
        super(Mtilinear_Net, self).__init__()
        self.ResTT = ResTT(lengthX=lengthX, lengthY=lengthY, kernel_size = 14, in_bond=in_bond, hidden_bond=hidden_bond, output_dim=out_dim,
                          channel = 1, std=0.01, name='ResTT')
        self.out_dim = out_dim

    def forward(self, x):
        x = self.ResTT(x)
        x = x.view(-1,self.out_dim)
        return x

class ResTTNet(nn.Module):
    def __init__(self, lengthX=14, lengthY=14,  in_bond=2, hidden_bond=2,out_dim = 10):
        super(ResTTNet, self).__init__()
        self.ResTT = ResTT(lengthX=lengthX, lengthY=lengthY, kernel_size = 14, in_bond=in_bond, hidden_bond=hidden_bond, output_dim=out_dim,
                          channel = 1, std=0.01, name='ResTT')
        self.out_dim = out_dim

    def forward(self, x):
        x = self.ResTT(x)
        x = x.view(-1,self.out_dim)
        return x

class TTNet(nn.Module):
    def __init__(self, lengthX=14, lengthY=14,  in_bond=2, hidden_bond=2,out_dim = 10, std = 0.01):
        super(TTNet, self).__init__()
        self.TT = TT(lengthX=lengthX, lengthY=lengthY, kernel_size = 14, in_bond=in_bond, hidden_bond=hidden_bond, output_dim=out_dim,
                          channel = 1, std=std, name='TT')
        self.out_dim = out_dim

    def forward(self, x):
        x = self.TT(x)
        x = x.view(-1,self.out_dim)
        return x

class TTBNNet(nn.Module):
    def __init__(self, lengthX=14, lengthY=14,  in_bond=2, hidden_bond=2,out_dim = 10):
        super(TTBNNet, self).__init__()
        self.TTBN = TTBN(lengthX=lengthX, lengthY=lengthY, kernel_size = 14, in_bond=in_bond, hidden_bond=hidden_bond, output_dim=out_dim,
                          channel = 1, std=0.01, name='TTBN')
        self.out_dim = out_dim

    def forward(self, x):
        x = self.TTBN(x)
        x = x.view(-1,self.out_dim)
        return x


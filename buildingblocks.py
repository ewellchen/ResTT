import torch
from torch import nn, Tensor
import numpy as np
from torch.autograd import Variable

class TT(nn.Module):

    def __init__(self, lengthX=14, lengthY=14, kernel_size = 14, in_bond=1, hidden_bond=1, output_dim=1, channel = 1, std=0.01, name='ResTT'):
        """
        Args:
            hidden_bond (int): dimension of hidden bond
            in_bond (int): dimension of input bond
            std (int): standard deviation
            output_dim (int): output dimension
            name (char): name of the model
        """
        super(TT, self).__init__()
        self.name = name
        self.channel = channel
        self.lengthX = lengthX
        self.lengthY = lengthY
        self.length = self.lengthX * self.lengthY
        self.in_bond = in_bond
        self.hidden_bond = hidden_bond
        self.std = std
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.weights1 = []
        self.weights2 = []
        self.weights3 = []
        self.output = []

        # definition of weight1
        w1 = nn.Parameter(torch.normal(torch.zeros([self.in_bond, self.hidden_bond, self.channel]), std=self.std))
        self.weights1.append(w1)
        for node in range(self.length - 2):
            w1 = nn.Parameter(torch.normal(torch.zeros([self.hidden_bond, self.in_bond, self.hidden_bond, self.channel]), std=self.std))
            self.weights1.append(w1)
        w1 = nn.Parameter(torch.normal(torch.zeros([self.hidden_bond, self.in_bond, self.output_dim, self.channel]), std=self.std))
        self.weights1.append(w1)
        self.weights1 = nn.ParameterList(self.weights1)

    def forward(self, x):
        self.InputBucket = []
        for k in range(self.lengthX - self.kernel_size + 1):
            for l in range(self.lengthY - self.kernel_size + 1):
                inputBucket = torch.reshape(x[:, :, k:k + self.kernel_size, l:l + self.kernel_size],
                                            (-1, self.in_bond, self.kernel_size * self.kernel_size))
                self.InputBucket.append(inputBucket)
        self.InputBucket = torch.stack(self.InputBucket, dim=3)
        y = torch.tensordot(self.InputBucket[:, :, 0, :], self.weights1[0], dims=([1], [0]))
        y = y.permute(0, 1, 3, 2) #nbrc->nbcr
        for i in range(1, self.lengthX * self.lengthY - 1):
            contra_w_d = torch.tensordot(self.InputBucket[:, :, i, :], self.weights1[i],
                                         dims=([1], [1]))  # nib,lirc->nblrc
            contra_w_d = contra_w_d.permute(0, 1, 4, 2, 3) #nblrc->nbclr
            y_new = torch.einsum('nbcr, nbcrl -> nbcl', y, contra_w_d) #nbcr * nbclr = nbcr
            y = y_new
        contra_w_d = torch.tensordot(self.InputBucket[:, :, self.lengthX * self.lengthY - 1, :],
                                     self.weights1[self.lengthX * self.lengthY - 1], dims=([1], [1]))  # nib,lioc->nbloc
        contra_w_d = contra_w_d.permute(0, 1, 4, 2, 3)  # nbloc->nbclo
        y_new = torch.einsum('nbcr, nbcrl -> nbcl', y, contra_w_d) #nbcr * nbclo = nbco
        out = y_new
        return out


class ResTT(nn.Module):
    def __init__(self, lengthX=14, lengthY=14, kernel_size = 14, in_bond=1, hidden_bond=1, output_dim=1, channel = 1, std=0.01, name='ResTT'):
        """
        Args:
            hidden_bond (int): dimension of hidden bond
            in_bond (int): dimension of input bond
            std (int): standard deviation
            output_dim (int): output dimension
            name (char): name of the model
        """
        super(ResTT, self).__init__()
        self.name = name
        self.channel = channel
        self.lengthX = lengthX
        self.lengthY = lengthY
        self.length = self.lengthX * self.lengthY
        self.in_bond = in_bond
        self.hidden_bond = hidden_bond
        self.std = std
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.weights1 = []
        self.weights2 = []
        self.weights3 = []
        self.output = []

        # definition of weight1
        w1 = nn.Parameter(torch.normal(torch.zeros([self.in_bond, self.hidden_bond, self.channel]), std=self.std))
        self.weights1.append(w1)
        for node in range(self.length - 2):
            w1 = nn.Parameter(torch.normal(torch.zeros([self.hidden_bond, self.in_bond, self.hidden_bond, self.channel]), std=self.std))
            self.weights1.append(w1)
        w1 = nn.Parameter(torch.normal(torch.zeros([self.hidden_bond, self.in_bond, self.output_dim, self.channel]), std=self.std))
        self.weights1.append(w1)
        self.weights1 = nn.ParameterList(self.weights1)

        # definition of weight2
        for node in range(self.length-1):
            w2 = nn.Parameter(torch.normal(torch.zeros([self.in_bond, self.hidden_bond, self.channel]), std=self.std))
            self.weights2.append(w2)
        w2 = nn.Parameter(torch.normal(torch.zeros([self.in_bond, self.output_dim, self.channel]), std=self.std))
        self.weights2.append(w2)
        self.weights2 = nn.ParameterList(self.weights2)

        # definition of weight3
        w3 = nn.Parameter(torch.normal(torch.zeros([self.hidden_bond, self.output_dim, self.channel]), std=self.std))
        self.weights3.append(w3)
        self.weights3 = nn.ParameterList(self.weights3)

    def forward(self, x):
        self.InputBucket = []
        for k in range(self.lengthX - self.kernel_size + 1):
            for l in range(self.lengthY - self.kernel_size + 1):
                inputBucket = torch.reshape(x[:, :, k:k + self.kernel_size, l:l + self.kernel_size],
                                            (-1, self.in_bond, self.kernel_size * self.kernel_size))
                self.InputBucket.append(inputBucket)
        self.InputBucket = torch.stack(self.InputBucket, dim=3)
        y = torch.tensordot(self.InputBucket[:, :, 0, :], self.weights1[0], dims=([1], [0]))
        y = y.permute(0, 1, 3, 2) #nbrc->nbcr
        for i in range(1, self.lengthX * self.lengthY - 1):
            contra_w_d = torch.tensordot(self.InputBucket[:, :, i, :], self.weights1[i],
                                         dims=([1], [1]))  # nib,lirc->nblrc
            contra_w_d = contra_w_d.permute(0, 1, 4, 2, 3) #nblrc->nbclr
            y_new = torch.einsum('nbcr, nbcrl -> nbcl', y, contra_w_d) #nbcr * nbclr = nbcr
            y_linear = torch.tensordot(self.InputBucket[:, :, i, :], self.weights2[i],
                                         dims=([1], [0]))  # nib,irc->nbrc
            y_linear = y_linear.permute(0, 1, 3, 2) # nbrc->nbcr
            y = y + y_linear + y_new
        contra_w_d = torch.tensordot(self.InputBucket[:, :, self.lengthX * self.lengthY - 1, :],
                                     self.weights1[self.lengthX * self.lengthY - 1], dims=([1], [1]))  # nib,lioc->nbloc
        contra_w_d = contra_w_d.permute(0, 1, 4, 2, 3)  # nbloc->nbclo
        y_new = torch.einsum('nbcr, nbcrl -> nbcl', y, contra_w_d) #nbcr * nbclo = nbco
        y_linear = torch.tensordot(self.InputBucket[:, :, self.lengthX * self.lengthY - 1, :],
                                   self.weights2[self.lengthX * self.lengthY - 1], dims=([1], [0]))  # nib,ioc->nboc
        y_linear = y_linear.permute(0, 1, 3, 2)  # nboc->nbco
        w3 = self.weights3[0].permute(2,0,1)  # loc->clo
        y =  torch.einsum('nbcr, cro -> nbco', y, w3)  # nbcr,clo->nbco
        out = y + y_linear + y_new
        return out

class TTBN(nn.Module):

    def __init__(self, lengthX=14, lengthY=14, kernel_size = 14, in_bond=1, hidden_bond=1, output_dim=1, channel = 1, std=0.01, name='ResTT'):
        """
        Args:
            hidden_bond (int): dimension of hidden bond
            in_bond (int): dimension of input bond
            std (int): standard deviation
            output_dim (int): output dimension
            name (char): name of the model
        """
        super(TTBN, self).__init__()
        self.name = name
        self.channel = channel
        self.lengthX = lengthX
        self.lengthY = lengthY
        self.length = self.lengthX * self.lengthY
        self.in_bond = in_bond
        self.hidden_bond = hidden_bond
        self.std = std
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.weights1 = []
        self.output = []
        self.bn = []

        # definition of weight1
        w1 = nn.Parameter(torch.normal(torch.zeros([self.in_bond, self.hidden_bond, self.channel]), std=self.std))
        bn = torch.nn.BatchNorm1d(self.hidden_bond)
        self.weights1.append(w1)
        self.bn.append(bn)
        for node in range(self.length - 2):
            w1 = nn.Parameter(torch.normal(torch.zeros([self.hidden_bond, self.in_bond, self.hidden_bond, self.channel]), std=self.std))
            self.weights1.append(w1)

        w1 = nn.Parameter(torch.normal(torch.zeros([self.hidden_bond, self.in_bond, self.output_dim, self.channel]), std=self.std))
        self.weights1.append(w1)
        self.weights1 = nn.ParameterList(self.weights1)
        self.bn = torch.nn.BatchNorm1d(self.hidden_bond)


    def forward(self, x):
        self.InputBucket = []
        for k in range(self.lengthX - self.kernel_size + 1):
            for l in range(self.lengthY - self.kernel_size + 1):
                inputBucket = torch.reshape(x[:, :, k:k + self.kernel_size, l:l + self.kernel_size],
                                            (-1, self.in_bond, self.kernel_size * self.kernel_size))
                self.InputBucket.append(inputBucket)
        self.InputBucket = torch.stack(self.InputBucket, dim=3)
        y = torch.tensordot(self.InputBucket[:, :, 0, :], self.weights1[0], dims=([1], [0]))
        y = y.permute(0, 1, 3, 2) #nbrc->nbcr
        for i in range(1, self.lengthX * self.lengthY - 1):
            contra_w_d = torch.tensordot(self.InputBucket[:, :, i, :], self.weights1[i],
                                         dims=([1], [1]))  # nib,lirc->nblrc
            contra_w_d = contra_w_d.permute(0, 1, 4, 2, 3) #nblrc->nbclr
            y_new = torch.einsum('nbcr, nbcrl -> ncbl', y, contra_w_d) #nbcr * nbclr = nbcr -> ncbr
            y = self.bn(y_new.view(-1,self.hidden_bond))
            y = y.view(-1, (self.lengthY - self.kernel_size + 1)**2,self.channel, self.hidden_bond)
        contra_w_d = torch.tensordot(self.InputBucket[:, :, self.lengthX * self.lengthY - 1, :],
                                     self.weights1[self.lengthX * self.lengthY - 1], dims=([1], [1]))  # nib,lioc->nbloc
        contra_w_d = contra_w_d.permute(0, 1, 4, 2, 3)  # nbloc->nbclo
        y_new = torch.einsum('nbcr, nbcrl -> nbcl', y, contra_w_d) #nbcr * nbclo = nbco
        out = y_new
        return out

if __name__ == '__main__':
    std = 0.01
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    x = torch.tensor(np.random.rand(100, 28, 28, 1), dtype=torch.float32)
    T1 = TTBN(kernel_size=3, in_bond=1, hidden_bond=2, channels=1, std=0.01, name='TT')(x).to(device)
    a = TTBN

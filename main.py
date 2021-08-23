# -*- coding: utf-8 -*-
"""Train the model"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import gzip
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from model import ResTTNet, TTNet, TTBNNet
from tools import mkdir, count_parameters, get_one_hot


# from loss import l1_loss,l2_loss


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 5 == 0:
            print('Train Epoch: {:3} [{:6}/{:6} ({:3.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item())
            )


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.MSELoss()(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=False)  # get the index of the max log-probability
            correct += pred.eq(target.argmax(dim=1, keepdim=False)).sum().item()

    test_loss /= len(test_loader)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss,
        correct,
        len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return 100. * correct / len(test_loader.dataset)
    # print('Accuracy:%f %%' % (100*correct/total))


class DealDataset(Dataset):
    """
        读取数据、初始化数据
    """

    def __init__(self, folder, data_name, label_name, transform=None, proportion=0.1, begin=5000):
        (image, train_labels) = load_data(folder, data_name, label_name)
        image = image[begin:begin + int(proportion * len(image))]
        train_labels = train_labels[begin:begin + int(proportion * len(train_labels))]
        image = (image - 255 / 2) / 255.
        train_set = []
        for i in range(len(image)):
            train_set.append(cv2.resize(np.reshape(image[i], [28, 28]), (14, 14),
                                        interpolation=cv2.INTER_CUBIC))
        train_set = np.expand_dims(train_set, axis=1)
        train_data = np.concatenate([np.cos(np.pi / 2 * train_set), np.sin(np.pi / 2 * train_set)], 1)
        self.train_set = train_data
        self.train_labels = get_one_hot(train_labels, 10)
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.train_set[index], self.train_labels[index]
        img = torch.tensor(img, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)
        return img, target

    def __len__(self):
        return len(self.train_set)


def load_data(data_folder, data_name, label_name):
    """
        data_folder: 文件目录
        data_name： 数据文件名
        label_name：标签数据文件名
    """
    with gzip.open(os.path.join(data_folder, label_name), 'rb') as lbpath:  # rb表示的是读取二进制数据
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(os.path.join(data_folder, data_name), 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)
    return x_train, y_train


if __name__ == '__main__':

    # parameters
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--test_batch_size', type=int, default=512, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--proportion', type=float, default=0.01, metavar='N',
                        help='proportion of training samples(default: 0.01)')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    repeat_time = 10
    for pro in [1]:
        # dataset
        trainDataset = DealDataset('data/MNIST/', "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                                   proportion=pro, begin=0)
        testDataset = DealDataset('data/MNIST/', "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz", proportion=1,
                                  begin=0)
        train_loader = DataLoader(dataset=trainDataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=testDataset, batch_size=args.test_batch_size, shuffle=True)
        Max_num = []
        name = ['proportion%.2f' % (pro)]
        for re in range(repeat_time):
            # model
            # model = TTNet(lengthX=14, lengthY=14,  in_bond=2, hidden_bond=50, out_dim=10, std = np.sqrt(1/50)).to(device)
            model = ResTTNet(lengthX=14, lengthY=14, in_bond=2, hidden_bond=50, out_dim=10).to(device)
            # model = TTBNNet(lengthX=14, lengthY=14, in_bond=2, hidden_bond=50, out_dim=10).to(device)
            # summary(model, input_size=(1, 28, 28))
            print('Total params:%d' % (count_parameters(model)))
            # training
            acc_test = []
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            for epoch in range(0, args.epochs):
                train(model, device, train_loader, optimizer, epoch)
                acc_test.append(test(model, device, test_loader))

            # saving
            mkdir('TT')
            Acc = np.reshape(np.array(acc_test), [1, len(acc_test)])
            Pd_data = pd.DataFrame(Acc, index=name)
            Pd_data.to_csv('./TT/F_accuracy_20.csv', mode='a', header=0)
            Max_num.append(Pd_data.iloc[0].max())
            torch.cuda.empty_cache()
        M_n = np.reshape(np.array(Max_num), [1, len(Max_num)])
        Pd_data1 = pd.DataFrame(M_n, index=name)
        Pd_data1.to_csv('./TT/F_result_20.csv', mode='a', header=0)
    # file_writer = open('result.txt', 'a')
    # file_writer.write('2TT_2FN' + '\n')
    # file_writer.write('test_acc:' + str(Pd_data.test.max()) + '\n')
    # file_writer.write('_________________________\n')
    # file_writer.flush()

    # if args.save_model:
    #     torch.save(model.state_dict(), "mnist_tnn.pt")

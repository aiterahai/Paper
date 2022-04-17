from model import VGGNet
from dataloader import dataloader
import torch
import torch.nn as nn
import torch.optim as optim

data_root = "../Datasets/CIFAR10"

class_number = 10
batch_size = 4
num_workers = 0
learning_rate = 1e-4
epoch = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

vggnet = VGGNet(class_number=class_number).to(device)

print(vggnet)

trainloader, testloader = dataloader(batch_size=batch_size,
                                     data_root=data_root,
                                     num_workers=num_workers)


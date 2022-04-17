from model import VGGNet
from dataloader import dataloader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

data_root = "../Datasets/CIFAR10"

class_number = 10
batch_size = 256
num_workers = 0
learning_rate = 1e-4
epoch = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

vggnet = VGGNet(class_number=class_number).to(device)

print(vggnet)

trainloader, testloader = dataloader(batch_size=batch_size,
                                     data_root=data_root,
                                     num_workers=num_workers)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(vggnet.parameters(), lr=learning_rate, momentum=0.9)

def train(epochs, model):
    for epoch_value in range(1, epochs + 1):
        total_loss = 0
        for img, data in tqdm(trainloader):
            img, data = img.to(device), data.to(device)

            optimizer.zero_grad()

            outputs = model(img)

            loss = criterion(outputs, data)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(f"EPOCH : {epoch_value} LOSS : {total_loss / len(trainloader)}")

train(epochs=epoch,
      model=vggnet)
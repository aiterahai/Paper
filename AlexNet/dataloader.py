import torch
from torchvision import datasets
from torchvision import transforms

def dataloader(batch_size,
               data_root,
               num_workers):

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.Dataloader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    testloader = torch.utils.data.Dataloader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return trainloader, testloader
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms

from matplotlib import pyplot as plt

is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class dataset:
    def __init__(self, data_path='../data', nr_tasks=10):
        self.data_path = data_path
        if not os.path.exists(data_path): os.makedirs(data_path)
        # Generate 10 incresing variances for noise addition
        self.task_noise = np.linspace(0, 1, num=nr_tasks)
    
    def get_data(self, task_id, batch_size):
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        transformations = [transforms.ToTensor(),
                           transforms.Normalize(mean,std),
                           AddGaussianNoise(0., float(self.task_noise[task_id]))]
        self.cifar_train = datasets.CIFAR10(self.data_path, train=True, download=True,
            transform=transforms.Compose(transformations))
        self.cifar_test = datasets.CIFAR10(self.data_path, train=False, download=True,
            transform=transforms.Compose(transformations))
        train_loader = torch.utils.data.DataLoader(self.cifar_train,
            batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(self.cifar_test,
            batch_size=batch_size, shuffle=False)
        return train_loader, test_loader
        

    #def get_data(self, batch_size):
    #    train_loader = torch.utils.data.DataLoader(self.mnist_train,
    #        batch_size=batch_size, shuffle=False)
    #    test_loader = torch.utils.data.DataLoader(self.mnist_test,
    #        batch_size=batch_size, shuffle=False)
    #    return train_loader, test_loader

def test_dataset():
    # Construct dataset
    # print the samples of dataset along with their label for each task.
    d = dataset()
    for t in range(10):
        t1_train, t1_test = d.get_data(t, 64)
        for local_batch, local_labels in t1_train:
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            plt.imshow(local_batch[0].permute(1,2,0))
            plt.show()
            break



if __name__ == "__main__":
    test_dataset()
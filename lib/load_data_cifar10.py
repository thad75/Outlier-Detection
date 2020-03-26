import os
import torch
import numpy as np
import torchvision.datasets as dset
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split 
from torch.utils.data import Subset
from torch.utils.data import DataLoader, ConcatDataset


def load_data_cifar10(batch_size, anomalous_class, train_size):

  training_set = dset.CIFAR10(root = './data', train = True, download=True, transform = transforms.ToTensor())    
  test_set = dset.CIFAR10(root = './data', train =False, download=True, transform = transforms.ToTensor())
  targets = torch.Tensor(training_set.targets)
  target_indices = np.arange(len(targets))

  # Split into train and train_bis
  train_idx_normal, training_idx_normal_anormal = train_test_split(target_indices, train_size=train_size)
  idx_to_keep = targets[train_idx_normal]!=anomalous_class
  idx_to_keep_a = targets[training_idx_normal_anormal]!=anomalous_class

  train_idx_normal = train_idx_normal[idx_to_keep]
  training_idx_normal_anormal = training_idx_normal_anormal[idx_to_keep_a]

  training_set_a = Subset(training_set, train_idx_normal)
  training_set_20 = Subset(training_set, training_idx_normal_anormal)

  test = ConcatDataset([training_set_20, test_set])

  train_loader = DataLoader(dataset=training_set_a, batch_size=batch_size, shuffle=True)
  test_loader = DataLoader(dataset = test, batch_size = 100, shuffle = False)
  
  return train_loader, test_loader
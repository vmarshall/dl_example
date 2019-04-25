
import os
from os import listdir,mkdir,rmdir
from os.path import join,isdir,isfile

import numpy as np

import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,utils
from torchvision.transforms import functional as func


class RandomRotate(object):
  def __call__(self, sample):
    X,Y = sample['X'],sample['Y']
    rotnum = np.random.choice(4)
    for ii in range(X.shape[0]):
      X[ii,:,:] = np.rot90(X[ii,:,:],k=rotnum,axes=(0,1))
    return {'X':X, 'Y':Y}

class RandomFlip(object):
  def __init__(self, flip_prob=0.5):
    self.flip_prob = flip_prob
  def __call__(self, sample):
    X,Y = sample['X'],sample['Y']
    if np.random.rand() > self.flip_prob:
      X[:,:,:] = X[:,:,::-1]
    return {'X':X, 'Y':Y}

class ToTensor(object):
  def __call__(self, sample):
    X,Y = sample['X'], sample['Y']
    return {'X': torch.from_numpy(X).float(), 'Y': torch.from_numpy(Y).long()}













class example_data(Dataset):
  def __init__(self, path_data, transform=None):
    self.path_data = path_data
    self.transform = transform
  
  def __len__(self):
    return len(listdir(self.path_data))
  
  def __getitem__(self,idx):
    X = np.zeros((1, 256, 256))
    Y = np.zeros((1, ))
    sample = {'X': X, 'Y': Y}
    if self.transform:
      sample = self.transform(sample)
    return sample

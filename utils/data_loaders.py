
import os
from os import listdir,mkdir,rmdir
from os.path import join,isdir,isfile

import pydicom
import itk
import numpy as np
from skimage.transform import resize

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

class Resize(object):
  def __init__(self, size=[256,256]):
    self.size = size
  def __call__(self, sample):
    X,Y = sample['X'],sample['Y']
    X_new = np.zeros((X.shape[0], self.size[0], self.size[1]))
    for ii in range(X.shape[0]):
      X_new[ii,:,:] = resize(X[ii,:,:], self.size, order=0, mode='constant', cval=0, anti_aliasing=True, preserve_range=True)
    return {'X':X_new, 'Y':Y}

class ToTensor(object):
  def __call__(self, sample):
    X,Y = sample['X'], sample['Y']
    return {'X': torch.from_numpy(X).float(), 'Y': torch.from_numpy(Y).long()}











def normalize_img(img):
  img = img.astype(np.float32)
  img -= np.min(img)
  img /= (np.max(img) + 1e-6)
  return img

def read_in_dcm_acqnum(path_dcm):
  dcm = pydicom.dcmread(path_dcm)
  img = dcm.pixel_array
  lab = (int(dcm.AcquisitionNumber) > 30) + 0
  if len(img.shape) == 3:
    img = img[:,:,0]
  img = normalize_img(img)
  return img,[lab]

def read_in_dcm_gender(path_dcm):
  dcm = pydicom.dcmread(path_dcm)

  try:
    PixelType = itk.ctype('signed short')
    Dimension = 2
    ImageType = itk.Image[PixelType, Dimension]
    reader = itk.ImageSeriesReader[ImageType].New()
    dicomIO = itk.GDCMImageIO.New()
    reader.SetImageIO(dicomIO)
    reader.SetFileNames([path_dcm])
    reader.Update()
    reader.GetOutput()
    img = itk.GetArrayFromImage(reader.GetOutput())
  except:
    img = np.zeros((256,256))

  lab = (dcm.PatientSex == 'M') + 0
  if len(img.shape) == 3:
    img = img[:,:,0]
  img = normalize_img(img)
  return img,[lab]

class example_data_front(Dataset):
  def __init__(self, path_data, ind=25, transform=None):
    self.path_data = path_data
    self.list_data = sorted(listdir(path_data))[:ind]
    self.transform = transform
  
  def __len__(self):
    return len(self.list_data)
  
  def __getitem__(self,idx):
    path_dcm = join(self.path_data, self.list_data[idx])
    X,Y = read_in_dcm_gender(path_dcm)
    X = X.reshape([1, X.shape[0], X.shape[1]])
    Y = np.array(Y).reshape([1, ])
    sample = {'X': X, 'Y': Y}
    if self.transform:
      sample = self.transform(sample)
    return sample

class example_data_back(Dataset):
  def __init__(self, path_data, ind=25, transform=None):
    self.path_data = path_data
    self.list_data = sorted(listdir(path_data))[ind:]
    self.transform = transform
  
  def __len__(self):
    return len(self.list_data)
  
  def __getitem__(self,idx):
    path_dcm = join(self.path_data, self.list_data[idx])
    X,Y = read_in_dcm_gender(path_dcm)
    X = X.reshape([1, X.shape[0], X.shape[1]])
    Y = np.array(Y).reshape([1, ])
    sample = {'X': X, 'Y': Y}
    if self.transform:
      sample = self.transform(sample)
    return sample


































class example_data(Dataset):
  def __init__(self, path_data,  transform=None):
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

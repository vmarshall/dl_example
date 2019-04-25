'''
Extremely bare bones wrapper class.
'''

import os
from os import listdir,mkdir,rmdir
from os.path import join,isdir,isfile
import shutil

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .architectures import *

class nnet_2d():
  def __init__(self, lr=1e-3, w_decay=1e-5, resume=False, load=False,
               path_save='/home/darvin/models', model_name='test',
               in_chan=3, class_num=2, net=GoogLeNet, gpu=True):
    self.lr         = lr #learning rate
    self.w_decay    = w_decay #L2 regularization
    self.path_chkpt = join(path_save, model_name+'_checkpoint.pth.tar')
    self.path_model = join(path_save, model_name+'_best.pth.tar')

    if gpu:
      self.net = nn.DataParallel(net(in_chan=in_chan, out_chan=class_num)).cuda()
    else:
      self.net = net(in_chan=in_chan, out_chan=class_num)
    self.loss = nn.CrossEntropyLoss()
    self.opt  = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.w_decay)

    self.best_loss = np.inf

    # Load in network if necessary.
    if resume:
      self.load_model(self.path_chkpt)
    elif load:
      self.load_model(self.path_model)

  def _compare_loss(self, new_loss):
    is_best = new_loss < self.best_loss
    if is_best:
      self.best_loss = new_loss
    return is_best

  def save_model(self, is_best=False, filename=None):
    state = {'net_dict': self.net.state_dict(),
             'opt_dict': self.opt.state_dict(),
             'best_loss': self.best_loss,
            }
    if filename is None:
      filename = self.path_chkpt
    torch.save(state, filename)
    if is_best:
      shutil.copyfile(filename, self.path_model)
    return 0

  def load_model(self, filename=None):
    if filename is None:
      filename = self.path_chkpt
    state = torch.load(filename)
    self.net.load_state_dict(state['net_dict'])
    self.opt.load_state_dict(state['opt_dict'])
    self.best_loss = state['best_loss']
    return 0

  def train_one_iter(self, X, Y):
    self.opt.zero_grad()
    self.net.train()
    pred = self.net(X)
    loss = self.loss(pred, Y)
    loss.backward()
    self.opt.step()
    return loss

  def val_one_iter(self, X, Y):
    self.net.eval()
    with torch.no_grad():
      pred = self.net(X)
      loss = self.loss(pred, Y)
      prob = F.softmax(pred, dim=1)
    return loss,prob

  def make_prediction(self, X):
    self.net.eval()
    with torch.no_grad():
      pred = self.net(X)
      prob = F.softmax(pred, dim=1)
    return prob

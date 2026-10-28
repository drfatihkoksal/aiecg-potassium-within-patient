"""ECGFounder fine-tuning helpers.

Vendored from PKUDigitalHealth/ECGFounder (MIT). One change: `pth` is now the
tensors-only checkpoint produced by ecgf_00_prepare_ckpt.py, so it loads with
weights_only=True. The upstream .pth stores a numpy scalar (val_auroc) that
torch >= 2.6 refuses under weights_only, and numpy 2.x module aliasing defeats
safe_globals; extracting the tensors once confines that unpickle to a single
deliberate step.
"""

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import json
from net1d import Net1D

import torch.nn as nn
import torch


def ft_12lead_ECGFounder(device, pth, n_classes, linear_prob=False):
  model = Net1D(
      in_channels=12, 
      base_filters=64, 
      ratio=1, 
      filter_list=[64,160,160,400,400,1024,1024],    
      m_blocks_list=[2,2,2,3,3,4,4], 
      kernel_size=16, 
      stride=2, 
      groups_width=16,
      verbose=False, 
      use_bn=False,
      use_do=False,
      n_classes=n_classes)

  state_dict = torch.load(pth, map_location=device, weights_only=True)

  state_dict = {k: v for k, v in state_dict.items() if not k.startswith('dense.')} 

  model.load_state_dict(state_dict, strict=False)

  model.dense = nn.Linear(model.dense.in_features, n_classes).to(device)
  # freezing model
  if linear_prob == True: 
    for name, param in model.named_parameters():
        if 'dense' not in name:  # no freezing last layer
            param.requires_grad = False

  model.to(device)

  return model


def ft_1lead_ECGFounder(device, pth, n_classes,linear_prob=False):
  model = Net1D(
      in_channels=1, 
      base_filters=64,
      ratio=1, 
      filter_list=[64,160,160,400,400,1024,1024],   
      m_blocks_list=[2,2,2,3,3,4,4],   
      kernel_size=16, 
      stride=2, 
      groups_width=16,
      verbose=False, 
      use_bn=False,
      use_do=False,
      n_classes=n_classes)

  state_dict = torch.load(pth, map_location=device, weights_only=True)

  state_dict = {k: v for k, v in state_dict.items() if not k.startswith('dense.')} 

  model.load_state_dict(state_dict, strict=False)

  model.dense = nn.Linear(model.dense.in_features, n_classes).to(device)

  if linear_prob == True: 
    for name, param in model.named_parameters():
        if 'dense' not in name:  # no freezing last layer
            param.requires_grad = False
  model.to(device)

  return model


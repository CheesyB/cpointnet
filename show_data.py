#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch
from torch.utils import data


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" id use_cuda else "cpu")
cudnn.benchmark = True

params = {'batch_size':1,
        'shuffle':True,
        'num_workers':6}




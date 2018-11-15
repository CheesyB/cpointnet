#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.utils.data as data
import torchvision.transforms as transforms
from pyntcloud import PyntCloud
from pcgen.util import utils 
import pandas as pd
import h5py
import numpy as np
import torch
import os
import os.path
import logging



class SceneDataset(data.Dataset):
    
    def __init__(self, hdf5_path, slices=100, npoints = 2500,neighbors_to_find=8000):
        
        self.logger = logging.getLogger('pointnet.SceneDataset')
        self._hdf5_path = hdf5_path
        if slices > 100:
            raise Exception('there are only 100 slices per scene')
        self._slices = slices  
        if npoints > neighbors_to_find:
            raise Exception('you can not sampel more points than you have neighbors')
        self._neighbors_to_find = neighbors_to_find
        self._npoints = npoints 
        self._raw_dataset = h5py.File(hdf5_path) 
        self._scenes = list(self._raw_dataset) # to make the dataset indexable 
        self._pointclouds = {} 
    
    def __len__(self):
        return self._slices * len(self._raw_dataset)
        
    
    def __getitem__(self, index):
        if index >= self.__len__():
            raise StopIteration
        scene_index, slice_index = divmod(index,self._slices)
        grp = self._raw_dataset[self._scenes[scene_index]]
        """ read the data from group """
        pc_slice = grp['slice{}'.format(slice_index)].value
        self.logger.info('getting part {}/{}'.format(index,len(self)))
        return (pc_slice[:,:3].astype(np.float32),pc_slice[:,3].astype(np.long))


        
if __name__ == '__main__':
    dataset = SceneDataset('data/dataneighbors.hd5f',5,2000)
    print(len(dataset))
    for idx,part in enumerate(dataset):
        utils.save_pointcloud_color(part,'data/tescht_cloud{}.ply'.format(idx))
        print('{}th sampel written'.format(idx))
        







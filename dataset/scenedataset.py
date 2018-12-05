#!/usr/bin/env python
# -*- coding: utf-8 -*- 

from pathlib import Path
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
        
import ipdb



class SceneDataset(data.Dataset):
    
    def __init__(self, hdf5_path, npoints = 2500):
        
        self.logger = logging.getLogger('pointnet.SceneDataset')
        filename = Path(hdf5_path)
        if not filename.is_file():
            raise Exception('huch the file is not there')
        self._hdf5_path = hdf5_path
        self._npoints = npoints
        self._raw_dataset = h5py.File(hdf5_path) 
        self._scenes = list(self._raw_dataset) # to make the dataset indexable 
        self._pointclouds = {} 
    
    def __len__(self):
        length = 0
        for grp_name in self._raw_dataset:
            grp = self._raw_dataset[grp_name]
            #ipdb.set_trace()
            length += grp.attrs['slices'] - 1
        return length
        
    
    def __getitem__(self, index):
        if index >= self.__len__():
            raise StopIteration
        scene_index,slice_index = self.get_indices(index)
        grp = self._raw_dataset[self._scenes[scene_index]]
        """ read the data from group """
        pc_slice = grp['slice{}'.format(slice_index)].value
        df = pd.DataFrame(pc_slice,columns=['x','y','z','class_number'])
        pc_slice = np.array(df.sample(self._npoints,axis=0))
        self.logger.info('getting part {}/{}'.format(index,len(self)))
        return (pc_slice[:,:3].astype(np.float32),pc_slice[:,3].astype(np.long))

    """ Ugly, do it with references in the hdf5 file """
    def get_indices(self,index):
        scene_index = 0
        slice_index = 0
        #ipdb.set_trace()
        for idx,grp_name in enumerate(self._scenes):
            grp = self._raw_dataset[grp_name] 
            index -= grp.attrs['slices']  - 1
            print(str(grp_name))
            print(str(grp))
            print(index)
            if index <= 0:
                slice_index = abs(index)
                break
            scene_index += 1
        print(index,scene_index, slice_index )
        return scene_index, slice_index 
             




if __name__ == '__main__':
    dataset = SceneDataset('data/new_dataset.hd5f')
    print(len(dataset))
    for idx,part in enumerate(dataset):
        utils.save_pointcloud_color(part,'data/tescht_cloud{}.ply'.format(idx))
        print('{}th sampel written'.format(idx))
        







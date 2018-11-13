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
    
    def __init__(self, hdf5_path, slices=100, npoints = 2500):
        
        self.logger = logging.getLogger('pointnet.SceneDataset')

        self._hdf5_path = hdf5_path
        self._slices = slices  
        self._npoints = npoints 
        #do stuff on attribute
        #load hdf5 file
        self._raw_dataset = h5py.File(hdf5_path) 
        self._scenes = list(self._raw_dataset)
        self._pointclouds = {} 
    
    def __len__(self):
        return self._slices * len(self._raw_dataset)
        
    
    def __getitem__(self, index):
        if index >= self.__len__():
            raise StopIteration
        scene_index, slice_index = divmod(index,self._slices)
        pc_tupel = self._get_pointcloud(scene_index)
        pc_slice = self._slice_scene(pc_tupel,scene_index,slice_index)
        return pc_slice 


    
    def _slice_scene(self,pc_tupel,scene_index,slice_index):
        pointcloud = pc_tupel[0]
        neighbors = pc_tupel[1]
        
        number_points = len(pointcloud.points)
        sample_point_index = int((number_points / self._slices ) * (slice_index) )
        
        points_sample = pointcloud.points.iloc[neighbors[sample_point_index]]
        less_points_sample = points_sample.sample(self._npoints)
        return np.array(less_points_sample)
   
    def _get_pointcloud(self,scene_index):
        if scene_index in self._pointclouds:
            return self._pointclouds[scene_index]
        print('preparing the {}th scene'.format(scene_index))
        grp = self._raw_dataset[self._scenes[scene_index]]
        raw_points = pd.DataFrame(grp['points4D'].value,columns=['x','y','z','class_number']) 
        pointcloud = PyntCloud(raw_points)
        neighbors = pointcloud.get_neighbors(k=8000)

        self._pointclouds[scene_index] = (pointcloud,neighbors)

        return (pointcloud,neighbors)



        
        
if __name__ == '__main__':
    dataset = SceneDataset('data/data.hd5f',5,2000)
    print(len(dataset))
    for idx,part in enumerate(dataset):
        utils.save_pointcloud_color(part,'data/tescht_cloud{}.ply'.format(idx))
        print('{}th sampel written'.format(idx))
        







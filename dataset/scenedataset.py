#!/usr/bin/env python
# -*- coding: utf-8 -*- 

from pathlib import Path
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
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
        tmp = self._raw_dataset['dataset'].items()
        tmp = list(iter(tmp))
        self._dataset = [np.array(item[1]) for item in tmp] #item[1] because item is a tuple 
                                                            # (ds_name,ds)

    def __len__(self):
        return len(self._dataset) 
        


    """ viel besser mit Referenzen  """
    def __getitem__(self, index):
        if index >= self.__len__():
            raise StopIteration
        current_slice = self._dataset[index] 
        df = pd.DataFrame(current_slice) # np -> pd -> np :))
        pc_slice = np.array(df.sample(self._npoints,axis=0))
        self.logger.info('getting part {}/{}'.format(index,len(self)))
        points = Variable(torch.from_numpy(pc_slice[:,:3].astype(np.float32))).cuda()
        target = Variable(torch.from_numpy(pc_slice[:,3].astype(np.long))).cuda()
        return (points,target)


if __name__ == '__main__':
    dataset = SceneDataset('data/current/new_format.hd5f')
    print(len(dataset))
    for idx,part in enumerate(dataset):
        utils.save_pointcloud_color(part,'data/tescht_cloud{}.ply'.format(idx))
        print('{}th sampel written'.format(idx))
        







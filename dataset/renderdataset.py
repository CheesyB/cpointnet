#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
import numpy as np
import pandas as pd
import h5py 
from pcgen.util import utils




""" This function assumes a hdf5 dataset from pcgen which has a nice structure
    and deals with references, which makes it a bit more complicated """
                                                                                                    
def render_dataset(file_path,save_path,scene='scene0'):
    dataset = h5py.File(file_path,'r')
    grp = dataset[scene]
    for idx,_slice in enumerate(grp):
        np_slice = grp[_slice].value
        """ das ist jetzt ein dummer hack, wegen den referencen in hdf5 """
        if isinstance(np_slice.all(),type(dataset.ref)): # mal schaun ob das so geht
            for idx,slice_ref in enumerate(np_slice):
                real_slice = dataset[slice_ref].value
                assert isinstance(real_slice, np.ndarray), 'the slice is still not an np.ndarray'
                utils.save_pointcloud_color(real_slice,save_path + '/' + 'slice' + str(idx) +'.ply')
            return                                                                                  
                                                                                                    
        utils.save_pointcloud_color(np_slice,save_path + '/' + str(_slice) + '.ply')      



if __name__ == "__main__":
    shutil.rmtree('delete',ignore_errors=True)
    os.mkdir('delete')
    render_dataset('tescht.hd5f','delete/')









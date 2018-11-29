#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import h5py 
from pcgen.util import utils




""" This function assumes a hdf5 dataset from pcgen with the 
    following structure:
    /-scene0
        |--> ds: 'cloud' (complete scene)
        |--> ds: 'slice0' 
        |--> ds: 'slice1' 
        |--> ds: 'slice2' 
        | ...
        |--> ds: 'slicen' 
      -scene1
        |--> ds: 'cloud' (complete scene)
        |--> ds: 'slice0' 
        |--> ds: 'slice1' 
        |--> ds: 'slice2' 
        | ...
        |--> ds: 'slicen' 
        ...
      -scenen
      and only writes the datasets from the first group """


def render_dataset(file_path,save_path):
    dataset = h5py.File(file_path,'r')
    grp = dataset['scene0']
    slices = []
    for idx,_slice in enumerate(grp):
        np_slice = grp[_slice].value
        utils.save_pointcloud_color(np_slice,save_path + str(_slice) + '.ply')
        
     



if __name__ == "__main__":
    render_dataset('data/new_dataset.hd5f','data/')

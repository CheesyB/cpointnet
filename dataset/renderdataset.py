#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import h5py 
from pcgen.util import utils






def render_dataset(file_path,save_path):

    dataset = h5py.File(file_path,'r')
    grp = dataset['scene0']
    slices = []
    for idx,_slice in enumerate(grp):
        np_slice = grp[_slice].value
        utils.save_pointcloud_color(np_slice,save_path + str(_slice) + '.ply')
        
     



if __name__ == "__main__":
    render_dataset('data/new_dataset.hd5f','data/')

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import glob
import os 
import numpy as np
import random
import time
import datetime
import torch
import torch.optim as optim
from torch.autograd import Variable
from pathlib import Path
from torch.autograd import Variable
from dataset.scenedataset import SceneDataset 
from pointnet import PointNetDenseCls
import torch.nn.functional as F
from pcgen.util import tictoc # brauchen wir das?
from pcgen.util import utils # brauchen wir das?
from  logger import Logger 
from dataset import renderdataset




    

def eval_segmentation(folder_path,params):
    
    logging.basicConfig(level=logging.DEBUG)                                                        
    logger = logging.getLogger('eval.segmentation')
    logging.getLogger('pointnet.SceneDataset').setLevel(level=logging.CRITICAL)

    """ where to save the net predictions """
    now = datetime.datetime.now()
    folder_path_eval = folder_path + '/eval_{}'.format(now.strftime('%d.%m_%H:%M'))
    os.makedirs(folder_path_eval,exist_ok=True)
    assert os.path.isdir(folder_path_eval) , 'folder path does not exist'
    logger.info('predicion output folder: {}'.format(folder_path_eval))
    
    """ clean setup """ 
    torch.cuda.empty_cache()
    manualSeed = random.randint(1, 10000) # fix seed
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    """ get dataset """
    batch_size = 20 
    eval_dataset = SceneDataset(params['testset_path'],2500)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=0)
    logger.info('length dataset: {}\n'
            'length training set: {}'.format(len(eval_dataset),len(eval_dataset)))

    """ render dataset """
    renderdataset.render_dataset(params['testset_path'],folder_path_eval)


    """ pararmeter """ 
    num_classes = params['number of classes']
    #num_batch = int(len(eval_dataset)/batch_size)
    num_batch = 1
    logger.info('We are looking for {} classes'.format(num_classes))


    """ setup net and load trained weights 
        search for the latest modified file which is the last model checkpoint """
    list_of_files = glob.glob(params['folder_path_chekp'] + '/*')
    latest_file = max(list_of_files, key=os.path.getctime)
    classifier = PointNetDenseCls(k = num_classes)
    classifier.load_state_dict(torch.load(latest_file))
    classifier.cuda()
    
    optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    
    """ for one epoch, why should we go for more:) """

    tick = time.time()
    for idx in range(num_batch):
        tack = time.time() - tick
        tick = time.time()
        _,data = next(enumerate(eval_dataloader, 0))
        points, target = data
        points = points.transpose(2,1) 
        
        """ tell classifier that its exam today"""
        classifier = classifier.eval()
        pred, _ = classifier(points)
        
        """ view is better than reshape (not sure) """ 
        pred = pred.view(-1, num_classes)
        target = target.view(-1,1)[:,0] - 1
        loss = F.nll_loss(pred, target)
        
        """ reshape until comparison is easy and render"""
        pred_choice = pred.data.max(1)[1]
        np_pred_choice = pred_choice.cpu().detach().numpy()
        np_points = data[0].cpu().detach().numpy()
        np_target = data[1].cpu().detach().numpy() - 1 
        utils.render_batch(np_points,np_pred_choice,
                folder_path_eval)
        utils.render_batch_bool(np_points,np_pred_choice,np_target,
                folder_path_eval)
        
        
        correct = pred_choice.eq(target.data).cpu().sum()
        accuracy = correct.item()/float(batch_size * 2500)

        """ tensorflow logger """ 
        info = { 'test_loss': loss.item(), 'test_accuracy': accuracy }
       
        """ console logger """
        logger.info('[{}: {}/{}] {} loss: {:2.3f} accuracy: {:2.3f}'.format(1, idx, 
            num_batch, 'test', loss.item(),accuracy))

if __name__ == "__main__":
    now = datetime.datetime.now()
    folder_path = '/home/tbreu/workbench/cpointnet/seg/08.12_18:08_run'
    params = {'testset_path':'dataset/data/current/test_C11_S15.hd5f',
            'folder_path_chekp':'/home/tbreu/workbench/cpointnet/seg/08.12_18:08_run/chekp',
            'number of classes':11,}
    eval_segmentation(folder_path,params)
        

#tensorsizes out of the net...
#torch.Size([32, 3, 2500])
#torch.Size([32, 2500])


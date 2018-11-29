#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os 
import numpy as np
import random
import time
import datetime
import torch
import torch.optim as optim
from pathlib import Path
from torch.autograd import Variable
from dataset.scenedataset import SceneDataset 
from pointnet import PointNetDenseCls
import torch.nn.functional as F
from pcgen.util import tictoc # brauchen wir das?
from pcgen.util import utils # brauchen wir das?
from  logger import Logger 


if __name__ == "__main__":
    
    
    logging.basicConfig(level=logging.DEBUG)                                                        
    logger = logging.getLogger('eval.segmentation')
    logging.getLogger('pointnet.SceneDataset').setLevel(level=logging.CRITICAL)

    """ where to save the net predictions """
    file_path = Path('out')
    suffix = len(list(file_path.iterdir()))
    os.mkdir(str(file_path) + '/pc_pred{}'.format(suffix))
    folder_path = str(file_path) + '/pc_pred{}'.format(suffix)
    logger.info('predicion output folder: {}'.format(folder_path))
    
    """ clean setup """ 
    torch.cuda.empty_cache()
    manualSeed = random.randint(1, 10000) # fix seed
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    """ get dataset """
    batch_size = 20
    eval_dataset = SceneDataset('dataset/data/evaldataset.hd5f',2500)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=0)
    logger.info('length dataset: {}\n'
            'length training set: {}'.format(len(eval_dataset),len(eval_dataset)))

    """ pararmeter """ 
    num_classes = 8
    num_batch = int(len(eval_dataset)/batch_size)
    num_batch = 1 # nur weil es noch kein datenset gibt
    logger.info('We are looking for {} classes'.format(num_classes))


    """ setup net and load trained weights """
    classifier = PointNetDenseCls(k = num_classes)
    classifier.load_state_dict(torch.load('seg/seg_model_99.pth'))
    classifier.cuda()
    optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    """ tensorflow hack from stackoverflow """ 
    tf_logger = Logger('./tflog')
    
    """ for one epoch, why should we go for more:) """

    tick = time.time()
    for idx in range(num_batch):
        tack = time.time() - tick
        tick = time.time()
        _,data = next(enumerate(eval_dataloader, 0))
        points, target = data
        points, target = Variable(points), Variable(target)
        points = points.transpose(2,1) 
        points, target = points.cuda(), target.cuda()
        
        """ tell classifier that its exam today"""
        classifier = classifier.eval()
        pred, _ = classifier(points)
        
        """ view is better than reshape (not sure) """ 
        pred = pred.view(-1, num_classes)
        target = target.view(-1,1)[:,0] - 1
        loss = F.nll_loss(pred, target)
        
        """ reshape until comparison is easy and render"""
        pred_choice = pred.data.max(1)[1]
        np_pred_choise = pred_choice.cpu().detach().numpy()
        utils.render_batch(np.array(data[0]),np_pred_choise,folder_path)
        
        correct = pred_choice.eq(target.data).cpu().sum()
        accuracy = correct.item()/float(batch_size * 2500)

        """ tensorflow logger """ 
        info = { 'test_loss': loss.item(), 'test_accuracy': accuracy }
        for tag, value in info.items():
            tf_logger.scalar_summary(tag, value, idx+1)
       
        """ console logger """
        logger.info('[{}: {}/{}] {} loss: {:2.3f} accuracy: {:2.3f}'.format(1, idx, 
            num_batch, 'test', loss.item(),accuracy))

        

#tensorsizes out of the net...
#torch.Size([32, 3, 2500])
#torch.Size([32, 2500])


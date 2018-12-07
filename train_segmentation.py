#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
import random
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from dataset.scenedataset import SceneDataset 
from eval_segmentation import eval_segmentation
from pointnet import PointNetDenseCls
import torch.nn.functional as F
from pcgen.util import tictoc
from  logger import Logger 


if __name__ == "__main__":
    global_tic = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=20, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--nepoch', type=int, default=250, help='number of epochs to train for')
    parser.add_argument('--outf', type=str, default='seg',  help='output folder')
    parser.add_argument('--model', type=str, default = '',  help='model path') 
    
    logging.basicConfig(level=logging.DEBUG)                                                        
    logger = logging.getLogger('train.segmentation')
    logging.getLogger('pointnet.SceneDataset').setLevel(level=logging.CRITICAL)
    time_logger = logging.getLogger('train.time')
    torch.cuda.empty_cache()
    
    opt = parser.parse_args()
    opt.manualSeed = random.randint(1, 10000) # fix seed
    
    
    now = datetime.datetime.now()
    params ={'created':now.strftime('%d.%m.%y %H:%M'),'manual seed':opt.manualSeed,
            'batch size':opt.batchSize,'epochs':opt.nepoch,'workers':opt.workers,
            'cont. training':opt.model}

    now = datetime.datetime.now()
    folder_path = 'seg/{}_run'.format(now.strftime('%d.%m_%H:%M'))
    params['folder_path'] = folder_path
    os.makedirs(folder_path,exist_ok=True)
    
    folder_path_tflogs = folder_path + '/tflog'
    params['folder_path_tflogs'] = folder_path_tflogs
    os.makedirs(folder_path_tflogs,exist_ok=True)
    
    folder_path_chekp = folder_path + '/chekp'
    params['folder_path_chekp'] = folder_path_chekp
    os.makedirs(folder_path_chekp,exist_ok=True)

    logger.info('{}'.format(opt))
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    params['trainset_path'] = 'dataset/hdf5/06-12_16:27_set/train_C11_S2.hd5f'
    dataset = SceneDataset(params['trainset_path'],2500)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                              shuffle=True, num_workers=int(opt.workers))
    
    params['testset_path'] = 'dataset/hdf5/06-12_16:27_set/train_C11_S2.hd5f'
    test_dataset = SceneDataset(params['testset_path'],2500)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                              shuffle=True, num_workers=int(opt.workers))

    params['len_dataset'] = len(dataset)
    params['len_testset'] = len(test_dataset)
    logger.info('length training set: {} '
            'length test set: {}'.format(params['len_dataset'],params['len_testset']))

    #num_classes = len(dataset.named_classe)
    num_classes = 11 
    params['number of classes'] = num_classes
    logger.info('We are looking for {} classes'.format(num_classes))


    classifier = PointNetDenseCls(k = num_classes)

    if opt.model != '':
        classifier.load_state_dict(torch.load(opt.model))

    params['learning rate'] = 0.01
    params['momentum'] = 0.9
    optimizer = optim.SGD(classifier.parameters(), 
            lr=params['learning rate'], momentum =  params['momentum'] )
    classifier.cuda()

    num_batch = int(len(dataset)/opt.batchSize)
    params['batches'] = num_batch
    
    tf_logger = Logger(folder_path_tflogs)
    tick = time.time()
    for epoch in range(opt.nepoch):
        for idx, data in enumerate(dataloader, 0):
            points, target = data
            points, target = Variable(points), Variable(target)
            points = points.transpose(2,1) 
            points, target = points.cuda(), target.cuda()   
            optimizer.zero_grad()
            classifier = classifier.train()
            pred, _ = classifier(points)
            pred = pred.view(-1, num_classes)
            target = target.view(-1,1)[:,0] -1 #warum -1??????? (keine 0 als Label!!!)
            loss = F.nll_loss(pred, target)
            loss.backward()
            optimizer.step()
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            accuracy = correct.item()/float(opt.batchSize * 2500)
            info = { 'train_loss': loss.item(), 'train_accuracy': accuracy }
            for tag, value in info.items():
                tf_logger.scalar_summary(tag, value, (epoch*num_batch)+(idx+1))
            
            logger.info('[{}: {}/{}] train loss: {:2.3f} accuracy: {:2.3f}'.format(epoch, idx, 
                num_batch, loss.item(),accuracy))
            
            if idx % 10 == 0:
                tack = time.time() - tick
                tick = time.time()
                _,data = next(enumerate(testdataloader, 0))
                points, target = data
                points, target = Variable(points), Variable(target)
                points = points.transpose(2,1) 
                points, target = points.cuda(), target.cuda()
                classifier = classifier.eval()
                pred, _ = classifier(points)
                
                pred = pred.view(-1, num_classes)
                target = target.view(-1,1)[:,0] - 1

                loss = F.nll_loss(pred, target)
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()
                accuracy = correct.item()/float(opt.batchSize * 2500)

                info = { 'test_loss': loss.item(), 'test_accuracy': accuracy }
                for tag, value in info.items():
                    tf_logger.scalar_summary(tag, value, (epoch*num_batch)+(idx+1))
                
                logger.info('[{}: {}/{}] {} loss: {:2.3f} accuracy: {:2.3f}'.format(epoch, idx, 
                    num_batch, 'test', loss.item(),accuracy))
                time_logger.info('Duration test: {0:.2f} |'
                        ' aprox. time left: {1}'.format(tack,
                            str(datetime.timedelta(seconds=int(tack*((opt.nepoch-epoch)\
                                *num_batch/10-idx/10))))))
        
        torch.save(classifier.state_dict(), folder_path_chekp + '/chekp{}.pth'.format(epoch))
    
    eval_segmentation(folder_path,params) 
    
    params['time'] = str(datetime.timedelta(seconds=(time.time() - global_tic)))
    with open(folder_path + '/info.json','x') as f: 
        json.dump(params,f)

        
#        for tag, value in model.named_parameters():
#            tag = tag.replace('.', '/')
#            logger.histo_summary(tag, value.data.cpu().numpy(), step+1)
#            logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), step+1)
#
#        # 3. Log training images (image summary)
#        info = { 'images': images.view(-1, 28, 28)[:10].cpu().numpy() }
#
#        for tag, images in info.items():
#        logger.image_summary(tag, images, step+1)


#torch.Size([32, 3, 2500])
#torch.Size([32, 2500])


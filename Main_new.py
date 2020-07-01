from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from datetime import datetime
import visdom

import argparse

import cv2
import os
from time import time

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

from networks.cenet import CE_Net_
from networks.unetpp import Unetpp
from networks.mru import MRU
from networks.cenet import Unet

from framework import MyFrame
from loss import dice_bce_loss,metrics
from data import ImageFolder
# from Visualizer import Visualizer
import torchvision

#from pytorchtools import EarlyStopping

import Constants
import image_utils

# Please specify the ID of graphics cards that you want to use
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
writer = SummaryWriter()



def main(config):

    #AME = 'CE-Net' + Constants.ROOT.split('/')[-1]

    # run the Visdom
    # viz = Visualizer(env=NAME)
    # Initialize the visualization environment

    print(config.model_type)


    if config.model_type == 'Unet':
        model = Unet
        save_img = 'img_u'
        save_mask = 'mask_u'
        save_pred = 'pred_u'

    if config.model_type == 'Unetpp':
        model = Unetpp
        save_img = 'img_upp'
        save_mask = 'mask_upp'
        save_pred = 'pred_upp'

    if config.model_type == "MRU":
        model = MRU
        save_img = 'img_mru'
        save_mask = 'mask_mru'
        save_pred = 'pred_mru'

    if config.model_type == 'CE_Net_':
        model = CE_Net_

        save_img = 'img'
        save_mask = 'mask'
        save_pred = 'pred'

    solver = MyFrame(model,dice_bce_loss , 2e-4)
    # batchsize = torch.cuda.device_count() * Constants.BATCHSIZE_PER_CARD

    batchsize = Constants.BATCHSIZE_PER_CARD
    batchsize_v = Constants.BATCH_VALID

    # For different 2D medical image segmentation tasks, please specify the dataset which you use
    # for examples: you could specify "dataset = 'DRIVE' " for retinal vessel detection.

    dataset = ImageFolder(root_path=Constants.ROOT, datasets='Brain',mode ='train')
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=4)


    valid = ImageFolder(root_path=Constants.ROOT, datasets='Brain', mode = 'valid')
    data_loader_v = torch.utils.data.DataLoader(
        valid,
        batch_size=batchsize_v,
        shuffle=True,
        num_workers=4)

    # start the logging files
    # mylog = open('logs/' + NAME + '.txt', 'w')
    tic = time()

    no_optim = 0
    total_epoch = Constants.TOTAL_EPOCH
    valid_epoch_best_dice_loss = 0

    #early_stopping = EarlyStopping(patience=20, verbose=True)

    for epoch in range(1, total_epoch + 1):
        data_loader_iter = iter(data_loader)
        train_epoch_loss = 0
        data_loader_iter_v = iter(data_loader_v)
        valid_epoch_loss = 0
        train_epoch_dice_loss = 0
        valid_epoch_dice_loss = 0
    

        for img, mask in data_loader_iter:
            solver.set_input(img, mask)
            train_loss, pred, train_dice_loss = solver.optimize()
            tsens,tspec,tacc,tprec = solver.eval() 
            train_epoch_loss += train_loss
            train_epoch_dice_loss += train_dice_loss

        b = 0       

        for img, mask in data_loader_iter_v:
            solver.set_input(img, mask)
            valid_loss, pred, valid_dice_loss = solver.optimize_test()
            sens,spec,acc,prec = solver.eval() 
            valid_epoch_loss += valid_loss
            valid_epoch_dice_loss += valid_dice_loss
        
        
            if( b % 20 == 0):

                # pred = pred.cpu().numpy()

                # mask = mask.cpu().numpy()

                # img = img.cpu().numpy() 

                # pred = pred[0].transpose(1,2,0)
                # img = img[0].transpose(1,2,0)
                # mask = mask[0].transpose(1,2,0)

                print('saving_pred_per_epoch_batch')

                # plt.imsave('valid/pred/' + str(epoch) + ' ' + str(b) +'.jpg', pred )

                # plt.imsave('valid/mask/' + str(epoch) + ' ' + str(b) + '.jpg', mask)

                # plt.imsave('valid/img/' + str(epoch) + ' ' + str(b) + '.jpg', img)
                
                torchvision.utils.save_image(img[0], "test/" + save_img +  "/" +str(epoch)+str(' ') + str(b) + ".jpg", nrow=1, padding=0) 
                torchvision.utils.save_image(mask[0], "test/" + save_mask + "/" +str(epoch)+str(' ') + str(b) + ".jpg", nrow=1, padding=0)
                torchvision.utils.save_image(pred[0], "test/" + save_pred + "/" +str(epoch)+str(' ') + str(b) + ".jpg", nrow=1, padding=0)

            b = b + 1

        
        # show the original images, predication and ground truth on the visdom.
        # show_image = (img + 1.6) / 3.2 * 255.
        # viz.img(name='images', img_=show_image[0, :, :, :])
        # viz.img(name='labels', img_=mask[0, :, :, :])
        # viz.img(name='prediction', img_=pred[0, :, :, :])

        train_epoch_loss = train_epoch_loss/len(data_loader_iter)
        train_epoch_dice_loss = train_epoch_dice_loss/len(data_loader_iter)
        valid_epoch_dice_loss = valid_epoch_dice_loss/len(data_loader_iter_v)
        valid_epoch_loss = valid_epoch_loss/len(data_loader_iter_v)


        # writer.add_scalars('Epoch_loss',{'train_epoch_loss': train_epoch_loss, 'train_epoch_dice_loss': train_epoch_dice_loss,
        # 'valid_epoch_loss':valid_epoch_loss,'valid_epoch_dice_loss':valid_epoch_dice_loss},epoch)

        writer.add_scalar('train_epoch_loss', train_epoch_loss, epoch)
        writer.add_scalar('train_epoch_dice_loss', train_epoch_dice_loss, epoch)
        writer.add_scalar('valid_epoch_loss',valid_epoch_loss , epoch)
        writer.add_scalar('valid_epoch_dice_loss', valid_epoch_dice_loss, epoch)



        # print("saving images")
        # print("length of (data_loader_iter) ", len(data_loader_iter))
        # print(mylog, '-----------------------------------------')
        # print(mylog, 'epoch:', epoch, '    time:', int(time() - tic))
        # print(mylog, 'train_loss:', train_epoch_loss)
        # print(mylog, 'valid_loss:', valid_epoch_loss)
        # print(mylog, 'train_dice_loss:', train_epoch_dice_loss)
        # print(mylog, 'valid_dice_loss:', valid_epoch_dice_loss)
        # print(mylog, 'sens:', sens)
        # print(mylog, 'SHAPE:', Constants.Image_size)
        print('in Training')
        print('epoch:', epoch, '    time:', int(time() - tic))
        print('train_loss:', train_epoch_loss)
        print('train_dice_loss', train_epoch_dice_loss)
        print('sensitivity:', tsens)
        print('specificity: ', tspec)
        print('accuracy:', tacc)
        print('precision:', tprec)

        # print('SHAPE:', Constants.Image_size)
        print('in VALIDATION')
        print('valid_loss:', valid_epoch_loss)
        print('valid_dice_loss', valid_epoch_dice_loss)
        print('sensitivity:', sens)
        print('specificity: ', spec)
        print('accuracy:', acc)
        print('precision:', prec)
        print('best_valid_dice_loss:',valid_epoch_best_dice_loss)
        print('--------------------------------------------------')

        if valid_epoch_best_dice_loss == 0:
            valid_epoch_best_dice_loss = valid_epoch_dice_loss

        elif valid_epoch_dice_loss >= valid_epoch_best_dice_loss:
            no_optim += 1

        elif valid_epoch_dice_loss < valid_epoch_best_dice_loss:
            no_optim = 0
            valid_epoch_best_dice_loss = valid_epoch_dice_loss
            solver.save('./weights/' + 'best' + config.model_type + '.th')

        elif no_optim > 20:
            #  print(mylog, 'early stop at %d epoch' % epoch)
             print('early stop at %d epoch' % epoch)
             break

        elif no_optim > Constants.NUM_UPDATE_LR:
            if solver.old_lr < 5e-7:
                break
            solver.save('./weights/' + 'best' + config.model_type + '.th')
            solver.update_lr(2.0, factor=True)

        # model = CE_Net_()

        # early_stopping(valid_epoch_loss, model)
        
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break
        # solver.save('./weights/' + 'best' + config.model_type + '.th')
        # mylog.flush()

    # print(mylog, 'Finish!')
    writer.close()
    print('Finish!')
    # mylog.close()


if __name__ == '__main__':
    print(torch.__version__)
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_type', type=str, default='CE_Net_', help='CE_Net_/MRU/Unetpp/Unet')

    config = parser.parse_args()

    main(config)
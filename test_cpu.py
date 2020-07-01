from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import cv2
import os
from time import time

import argparse

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

from networks.cenet import CE_Net_
from framework import MyFrame
from loss import dice_bce_loss,metrics
from data import ImageFolder
from Visualizer import Visualizer
import torchvision

import Constants
import image_utils



# Please specify the ID of graphics cards that you want to use
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def CE_Net_Train():
    # NAME = 'CE-Net' + Constants.ROOT.split('/')[-1]

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

    # run the Visdom
    # viz = Visualizer(env=NAME)

    solver = MyFrame(CE_Net_, dice_bce_loss, 2e-4,metrics)
    print("loading the weights")
    solver.load('./weights/' + 'best.th')
    solver.load('./weights/' + 'best' + config.model_type + '.th')
    # batchsize = torch.cuda.device_count() * Constants.BATCHSIZE_PER_CARD

    batchsize_v = Constants.BATCH_VALID

    # For different 2D medical image segmentation tasks, please specify the dataset which you use
    # for examples: you could specify "dataset = 'DRIVE' " for retinal vessel detection.

    valid = ImageFolder(root_path=Constants.ROOT, datasets='Brain', mode = 'valid')
    data_loader_v = torch.utils.data.DataLoader(
        valid,
        batch_size=batchsize_v,
        shuffle=True,
        num_workers=4)


    total_epoch = 1


    for epoch in range(1, total_epoch + 1):
        a = 0
        data_loader_iter_v = iter(data_loader_v)
        valid_epoch_loss = 0
        valid_epoch_dice_loss = 0

        for img, mask in data_loader_iter_v:
            solver.set_input(img, mask)
            valid_loss, pred, valid_dice_loss = solver.optimize_test()
            sens,spec,acc,prec = solver.eval()
            valid_epoch_loss += valid_loss
            valid_epoch_dice_loss += valid_dice_loss

            print('saving_pred_mask')

            torchvision.utils.save_image(pred[0], "test/" + save_pred + "/" +str(epoch)+str(' ') + str(b) + ".jpg", nrow=1, padding=0)

            a = a + 1

        valid_epoch_dice_loss = valid_epoch_dice_loss/len(data_loader_iter_v)
        valid_epoch_loss = valid_epoch_loss/len(data_loader_iter_v)

        print('test_loss:', valid_epoch_loss)
        print('test_dice_loss', valid_epoch_dice_loss)
        print('sensitivity:', sens)
        print('specificity: ', spec)
        print('accuracy:', acc)
        print('precision:', prec)

        # print('sensitivity:', sens)
        print('---------------------------------------------')



    print('Finish!')


if __name__ == '__main__':
    print(torch.__version__)
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_type', type=str, default='CE_Net_', help='CE_Net_/MRU/Unetpp/Unet')

    config = parser.parse_args()

    main(config)
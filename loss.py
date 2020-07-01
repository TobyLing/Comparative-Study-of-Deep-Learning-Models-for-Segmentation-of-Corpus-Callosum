import torch
import torch.nn as nn
from torch.autograd import Variable as V

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import cv2
import numpy

import skimage as ski 
from sklearn.metrics import confusion_matrix
import numpy as np

class weighted_cross_entropy(nn.Module):
    def __init__(self, num_classes=3, batch=True):
        super(weighted_cross_entropy, self).__init__()
        self.batch = batch
        self.weight = torch.Tensor([52.] * num_classes).cuda()
        self.ce_loss = nn.CrossEntropyLoss(weight=self.weight)

    def __call__(self, y_true, y_pred):

        y_ce_true = y_true.squeeze(dim=1).long()


        a = self.ce_loss(y_pred, y_ce_true)

        return a


class dice_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_loss, self).__init__()
        self.batch = batch

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(0)
            j = y_pred.sum(1).sum(1).sum(0)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(0)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_true, y_pred):

        b = self.soft_dice_loss(y_true, y_pred)
        return b



    def test_weight_cross_entropy():
        N = 4
        C = 12
        H, W = 128, 128

        inputs = torch.rand(N, C, H, W)
        targets = torch.LongTensor(N, H, W).random_(C)
        inputs_fl = Variable(inputs.clone(), requires_grad=True)
        targets_fl = Variable(targets.clone())
        print(weighted_cross_entropy()(targets_fl, inputs_fl))


class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_true, y_pred):
        a = self.bce_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_true, y_pred)
        return a,b


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N, H, W = target.size(0), target.size(2), target.size(3)
        smooth = 1

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss


class MulticlassDiceLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """

    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()

    def forward(self, input, target, weights=None):

        C = target.shape[1]

        # if weights is None:
        # weights = torch.ones(C) #uniform weights for all classes

        dice = DiceLoss()
        totalLoss = 0

        for i in range(C):
            diceLoss = dice(input[:, i, :, :], target[:, i,:, :])
            if weights is not None:
                diceLoss *= weights[i]
            totalLoss += diceLoss

        return totalLoss

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, target, input):
        target1 = torch.squeeze(target, dim=1)
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target2 = target1.view(-1,1).long()

        logpt = F.log_softmax(input, dim=1)
        # print(logpt.size())
        # print(target2.size())
        logpt = logpt.gather(1,target2)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


class metrics:
    def __init__(self,y_true,y_pred):


        # y_predb = ski.img_as_bool(y_pred).astype(int)
        # print(y_true.unique())
        # print(y_pred.unique())

        self.tp = torch.sum(y_true * y_pred)
        self.tn = torch.sum((1-y_true) * (1-y_pred) )
        self.fp = torch.sum( (1-y_true) * (y_pred))
        self.fn = torch.sum( y_true * (1-y_pred))


        # self.tn, self.fp, self.fn, self.tp = confusion_matrix(y_true,y_pred).ravel()

        # print(confusion_matrix(y_true,y_pred))

        # for k in range(y_true.shape[0]):
        #     for i in range(y_true.shape[2]):
        #         for j in range(y_pred.shape[2]):
        #             if (y_true[k,:,i,j] == 1):
        #                 if (y_pred[k,:,i,j] == 0):
        #                     self.fn = self.fn + 1 

        # for k in range(y_true.shape[0]):
        #     for i in range(y_true.shape[2]):
        #         for j in range(y_pred.shape[2]):
        #             if (y_true[k,:,i,j] == 0):
        #                 if (y_pred[k,:,i,j] == 1):
        #                     self.fp = self.fp + 1 

        # for k in range(y_true.shape[0]):
        #     for i in range(y_true.shape[2]):
        #         for j in range(y_pred.shape[2]):
        #             if (y_true[k,:,i,j] == 0):
        #                 if (y_pred[k,:,i,j] == 0):
        #                     self.tn = self.tn + 1 


    def sensitivity(self):
        gt = self.tp + self.fn 
        if gt == 0:
            return 1
        else:
            sens = self.tp / gt 
            return sens

    def accuracy(self):
        acc = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)
        return acc 

    def specificity(self):
        spec =  self.tn / (self.fp + self.tn) 
        return spec

    def precision(self):
        prec = self.tp / (self.tp + self.fp)
        return prec 


    # def __call__(self, y_true, y_pred):
    #     a = self.sensitivity()
    #     return a 


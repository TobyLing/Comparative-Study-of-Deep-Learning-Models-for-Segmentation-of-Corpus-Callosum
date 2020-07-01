import torch
import torch.nn as nn
from torch.autograd import Variable as V
from loss import metrics 

import cv2
import numpy as np


class MyFrame():
    def __init__(self, net, loss, lr=2e-4, evalmode=False):
        self.net = net().cuda()
        #self.net = net()
        #self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)
        # self.optimizer = torch.optim.SGD(params=self.net.parameters(), lr=lr)
        self.loss = loss()
        self.old_lr = lr
        if evalmode:
            for i in self.net.modules():
                if isinstance(i, nn.BatchNorm2d):
                    i.eval()
        
    def set_input(self, img_batch, mask_batch=None, img_id=None):
        self.img = img_batch
        self.mask = mask_batch
        self.img_id = img_id
        
    def test_one_img(self, img):
        pred = self.net.forward(img)
        
        pred[pred>0.5] = 1
        pred[pred<=0.5] = 0

        mask = pred.squeeze().cpu().data.numpy()
        return mask
    
    def test_batch(self):
        self.forward(volatile=True)
        mask =  self.net.forward(self.img).cpu().data.numpy().squeeze(1)
        mask[mask>0.5] = 1
        mask[mask<=0.5] = 0
        
        return mask, self.img_id
    
    def test_one_img_from_path(self, path):
        img = cv2.imread(path)
        img = np.array(img, np.float32)/255.0 * 3.2 - 1.6
        img = V(torch.Tensor(img).cuda())
        
        mask = self.net.forward(img).squeeze().cpu().data.numpy()#.squeeze(1)
        mask[mask>0.5] = 1
        mask[mask<=0.5] = 0
        
        return mask
        
    def forward(self, volatile=False):
        self.img = V(self.img.cuda(), volatile=volatile)
        # self.img = V(self.img, volatile=volatile)
        if self.mask is not None:
            self.mask = V(self.mask.cuda(), volatile=volatile)
            # self.mask = V(self.mask, volatile=volatile)

    def optimize(self):
        self.forward()
        self.optimizer.zero_grad()
        pred = self.net.forward(self.img)
        loss,dice_loss = self.loss(self.mask, pred)
        loss.backward()
        self.optimizer.step()
        return loss.data, pred,dice_loss

    def optimize_test(self):
        with torch.no_grad():
            self.forward()
            pred = self.net.forward(self.img)
            loss,dice = self.loss(self.mask, pred)
            # loss.backward()
            # self.optimizer.step()
        return loss.data, pred,dice
        
    def save(self, path):
        torch.save(self.net.state_dict(), path)
        
    def load(self, path):
        device = torch.device("cuda")
        # device = torch.device("cpu")
        # self.net = nn.DataParallel(self.net)
        # self.net.load_state_dict(torch.load(path,map_location= "cpu" ))
        self.net.load_state_dict(torch.load(path))
        self.net.to(device)
    
    def update_lr(self, new_lr, factor=False):
        if factor:
            new_lr = self.old_lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        #print (mylog, 'update learning rate: %f -> %f' % (self.old_lr, new_lr))
        print ('update learning rate: %f -> %f' % (self.old_lr, new_lr))
        self.old_lr = new_lr

        
    def eval(self):
        self.forward()
        self.optimizer.zero_grad()
        pred = self.net.forward(self.img)
        met = metrics(self.mask, pred)
        a = met.sensitivity()
        b = met.specificity()
        c = met.accuracy()
        d = met.precision()

        # m = torch.mean(pred)
        # pred[pred>=m] = 1
        # pred[pred<m] = 0
        # inter = torch.sum( pred * self.mask)
        # summ = torch.sum(pred) + torch.sum(self.mask)
        # dice = 2 * inter / (summ)    
        return a,b,c,d

    
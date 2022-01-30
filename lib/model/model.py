"""GANomaly
"""
# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
import os
import time
import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from lib.model.basemodel import BaseModel
from lib.utils.loss_functions import l2_loss, MseDirectionLoss
from lib.model.networks import define_G, define_D, get_scheduler




class DeepDisaster(BaseModel):
    """DeepDisaster Class
    """

    def __init__(self, opt, data=None, teacher=True):
        super(DeepDisaster, self).__init__(opt, data, teacher)
        ##

        # -- Misc attributes
        self.add_noise = True
        self.epoch = 0
        self.times = []
        self.total_steps = 0
        self.teacher = teacher
        self.f_x_hat_T = None
        self.x_hat_T = None

        self.err_g_kd = 0
        self.err_d_kd = 0

        ##
        # Create and initialize networks.
        self.netg = define_G(self.opt, norm='batch', use_dropout=False, init_type='normal', teacher=self.teacher)
        self.netd = define_D(self.opt, norm='batch', use_sigmoid=False, init_type='normal', teacher=self.teacher)


        if self.teacher:
            print("Load Pre-Trained Teacher...")
            self.nz = self.opt.nz_T
            #add teacher path pretrained!
            self.netg.load_state_dict(torch.load('pre_trained/imagenet_netG_last_new.pth')['state_dict'])
            self.netd.load_state_dict(torch.load('pre_trained/imagenet_netD_last_new.pth')['state_dict'])
            print("\tDone.\n")
        ##
        if self.opt.resume != '':
            if not self.teacher: 
                print("Loading Pre-trained Student...")
                self.nz = self.opt.nz_S
                self.opt.iter = torch.load(os.path.join(self.opt.resume, 'kd_netG_best.pth'))['epoch']
                self.netg.load_state_dict(torch.load(os.path.join(self.opt.resume, 'kd_netG_best.pth'))['state_dict'])
                self.netd.load_state_dict(torch.load(os.path.join(self.opt.resume, 'kd_netD_best.pth'))['state_dict'])
                print("\tDone.\n")


        if self.opt.load_weights:
            if not self.teacher: 
                self.load_weights(is_best=True, path = self.opt.checkpoints_path)

        if self.opt.verbose:
            print(self.netg)
            print(self.netd)

        ##
        # Loss Functions
        self.l_adv = nn.BCELoss()
        self.l_con = nn.L1Loss()
        self.l_lat = l2_loss
        self.kd_loss = MseDirectionLoss(self.opt.lambdaa)

        ##
        # Initialize input tensors.
        self.gt = torch.empty(size=(opt.batchsize,), dtype=torch.long, device=self.device)
        self.label = torch.empty(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.real_label = torch.ones (size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.fake_label = torch.zeros(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)

        self.fixed_input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)
        self.noise = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)
        self.input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)

        ##
        # Setup optimizer
        if self.opt.isTrain:
            self.netg.train()
            self.netd.train()
            self.optimizers  = []
            self.optimizer_g = optim.Adam(self.netg.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizer_d = optim.Adam(self.netd.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_d)
            self.optimizers.append(self.optimizer_g)

            self.schedulers = [get_scheduler(optimizer, opt) for optimizer in self.optimizers]


 
    def forward(self):
        self.forward_g()
        self.forward_d()

    def forward_g(self):
        """ Forward propagate through netG
        """
        self.fake = self.netg(self.input + self.noise)
     

    def forward_d(self):
        """ Forward propagate through netD
        """
        self.pred_real, self.feat_real = self.netd(self.input)
        self.pred_fake, self.feat_fake = self.netd(self.fake)
      
    def backward_g(self):
        """ Backpropagate netg
        """
        self.err_g_adv = self.opt.w_adv * self.l_adv(self.pred_fake, self.real_label)
        self.err_g_con = self.opt.w_con * self.l_con(self.fake, self.input)
        self.err_g_lat = self.opt.w_lat * self.l_lat(self.feat_fake, self.feat_real)
        
        # KD loss 
        if not self.teacher:
            self.err_g_kd = self.opt.w_kg * self.kd_loss(self.x_hat_T , self.fake)
            self.err_d_kd = self.opt.w_kd* self.kd_loss(self.f_x_hat_T , self.feat_fake)
       
        self.err_g = self.err_g_adv + self.err_g_con + self.err_g_lat + self.err_g_kd + self.err_d_kd
        
        self.err_g.backward(retain_graph=True)

    def backward_d(self):
        # Fake
        pred_fake, _ = self.netd(self.fake.detach())
        self.err_d_fake = self.l_adv(pred_fake, self.fake_label)

        # Real
        self.err_d_real = self.l_adv(self.pred_real, self.real_label)
        
        # Combine losses.
        self.err_d = self.err_d_real + self.err_d_fake + self.err_g_lat + self.err_d_kd
        self.err_d.backward(retain_graph=True)

    def update_netg(self):
        """ Update Generator Network.
        """       
        self.optimizer_g.zero_grad()
        self.backward_g()
        self.optimizer_g.step()

    def update_netd(self):
        """ Update Discriminator Network.
        """       
        self.optimizer_d.zero_grad()
        self.backward_d()
        self.optimizer_d.step()
        if self.err_d < 1e-5: self.reinit_d()
    ##
    def optimize_params(self, f_x_hat_T=None, x_hat_T=None):
        """ Optimize netD and netG  networks.
        """
        self.f_x_hat_T = f_x_hat_T
        self.x_hat_T =x_hat_T        
        self.forward()
        self.update_netg()
        self.update_netd()
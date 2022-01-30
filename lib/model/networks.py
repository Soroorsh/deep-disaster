""" Network architectures.
"""
# pylint: disable=W0221,W0622,C0103,R0913

##
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel

from torch.nn import init
from torch.optim import lr_scheduler



###############################################################################
# Helper Functions
###############################################################################
##
def get_norm_layer(norm_type='instance'):
    """
    defines normalization function based on the norm_type input

    Args:
        norm_type (str): type of normalization layer : 'none'| 'batch' | 'instance' 
        
    Returns:
         Normalization layer funtion
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

##
def get_scheduler(optimizer, opt):
    """
    defines scheduler function based on the opt.lr_policy

    Args:
        optimizer: defined optimizer 
        opt  : input config
    Returns:
         the scheduler
    """
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.iter - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

##
def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)

##
def init_net(net, init_type='normal', gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type)
    return net

##
def define_G(opt, norm='batch', use_dropout=False, init_type='normal', teacher=True):
    netG = None
    norm_layer = get_norm_layer(norm_type=norm)
    num_layer = int(np.log2(opt.isize))
    if teacher:
        netG = UnetGenerator(opt.nc, opt.nc, num_layer, opt.ngf_T, norm_layer=norm_layer, use_dropout=use_dropout,teacher=teacher)
        return init_net(netG, init_type, opt.gpu_ids)
    elif not teacher:
        netG = UnetGenerator(opt.nc, opt.nc, num_layer, opt.ngf_S, norm_layer=norm_layer, use_dropout=use_dropout,teacher=teacher)
        return init_net(netG, init_type, opt.gpu_ids)

##
def define_D(opt, norm='batch', use_sigmoid=False, init_type='normal' , teacher=True):
    netD = None
    norm_layer = get_norm_layer(norm_type=norm)
    netD = BasicDiscriminator(opt,teacher)
    return init_net(netD, init_type, opt.gpu_ids)


##############################################################################
# Classes
##############################################################################


##
class BasicDiscriminator(nn.Module):
    """
    NETD
    """
    def __init__(self, opt, teacher):
        super(BasicDiscriminator, self).__init__()
        isize = opt.isize

        if teacher: 
            nz = opt.nz_T
            nc = opt.nc
            ngf = opt.ngf_T
            ndf = opt.ndf_T
        else: 
            nz = opt.nz_S
            nc = opt.nc
            ngf = opt.ngf_S
            ndf = opt.ndf_S

        n_extra_layers = 0
        self.ngpu = opt.ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        feat = nn.Sequential()
        clas = nn.Sequential()
        # input is nc x isize x isize
        feat.add_module('initial-conv-{0}-{1}'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        feat.add_module('initial-relu-{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            feat.add_module('extra-layers-{0}-{1}-conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            # if teacher:
            feat.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, cndf),
                                nn.BatchNorm2d(cndf))
            feat.add_module('extra-layers-{0}-{1}-relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            feat.add_module('pyramid-{0}-{1}-conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            feat.add_module('pyramid-{0}-batchnorm'.format(out_feat),
                                nn.BatchNorm2d(out_feat))
            feat.add_module('pyramid-{0}-relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        # main.add_module('final-{0}-{1}-conv'.format(cndf, 1),
        #                     nn.Conv2d(cndf, nz, 4, 1, 0, bias=False))
        feat.add_module('final-{0}-{1}-conv'.format(cndf, 1),
                            nn.Conv2d(cndf, nz, 4, 1, 0, bias=False))
        clas.add_module('classifier', nn.Conv2d(nz, 1, 3, 1, 1, bias=False))
        clas.add_module('Sigmoid', nn.Sigmoid())

        self.feat = feat
        self.clas = clas

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            feat = nn.parallel.data_parallel(self.feat, input, range(self.ngpu))
            clas = nn.parallel.data_parallel(self.clas, feat, range(self.ngpu))
        else:
            feat =  self.feat(input)
            clas = self.clas(feat)
        clas = clas.view(-1, 1).squeeze(1)
        return clas, feat


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, teacher=True):
        super(UnetGenerator, self).__init__()

          # placeholder for the gradients
        self.gradients = None
        self.activation = None

        
        self.unet_block_innermost = None
        self.unet_block = None
        self.unet_block_1 = None
        self.unet_block_2 = None
        self.unet_block_3 = None
        self.unet_block_4 = None


        # construct unet structure
        self.unet_block_innermost = UnetSkipConnectionBlock(512, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, teacher=teacher)
        for i in range(num_downs - 5):
            if i == 0:
                self.unet_block = UnetSkipConnectionBlock(ngf * 8, 512, input_nc=None, submodule=self.unet_block_innermost, norm_layer=norm_layer, use_dropout=use_dropout, teacher=teacher)
            else:   
                self.unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=self.unet_block, norm_layer=norm_layer, use_dropout=use_dropout, teacher=teacher)
        self.unet_block_1 = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=self.unet_block, norm_layer=norm_layer, teacher=teacher)
        self.unet_block_2 = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=self.unet_block_1, norm_layer=norm_layer, teacher=teacher)
        self.unet_block_3 = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=self.unet_block_2, norm_layer=norm_layer, teacher=teacher)
        self.unet_block_4 = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=self.unet_block_3, outermost=True, norm_layer=norm_layer, teacher=teacher)

        self.model = self.unet_block_4

    
    def activations_hook(self, grad):
            self.gradients = grad
            
    def forward(self, input):
        
        x = self.model(input)
        if input.requires_grad:
            self.activation = x
            h = x.register_hook(self.activations_hook)
        return x


    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.activation




# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, teacher=True):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
          
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
           
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
          
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
         
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)

        else:
            return torch.cat([x, self.model(x)], 1)

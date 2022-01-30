""" BaseModel
"""
# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
import os
import time
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import torchvision.utils as vutils

from tqdm import tqdm
from collections import OrderedDict
from torch.autograd import Variable
from sklearn.metrics import roc_curve, auc

from lib.utils.loss_functions import l2_loss
from lib.utils.visualizer import Visualizer
from lib.model.networks import define_G, define_D, get_scheduler


class BaseModel():
    """ Base Model for ganomaly
    """
    def __init__(self, opt, data, teacher=True):
        ##
        # Seed for deterministic behavior
        self.seed(opt.manualseed)

        # Initalize variables.
        self.opt = opt
        self.visualizer = Visualizer(opt)
        self.data = data
        self.device = torch.device("cuda:{}".format(self.opt.gpu_ids[0]) if self.opt.device != 'cpu' else "cpu")
        self.teacher = teacher

        if self.teacher: 
            self.nz = self.opt.nz_T
        elif not self.teacher: 
            self.nz = self.opt.nz_S
    ##
    def seed(self, seed_value):
        """ Seed 

        Arguments:
            seed_value {int} -- [description]
        """
        # Check if seed is default value
        if seed_value == -1:
            return

        # Otherwise seed all functionality
        import random
        random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        np.random.seed(seed_value)
        torch.backends.cudnn.deterministic = True

    ##
    def set_input(self, input:torch.Tensor, noise:bool=False):
        """ Set input and ground truth

        Args:
            input (FloatTensor): Input data for batch i.
        """
        with torch.no_grad():
            self.input.resize_(input[0].size()).copy_(input[0])
            self.gt.resize_(input[1].size()).copy_(input[1])
            self.label.resize_(input[1].size())

            # Add noise to the input.
            if noise: self.noise.data.copy_(torch.randn(self.noise.size()))

            # Copy the first batch as the fixed input.
            if self.total_steps == self.opt.batchsize:
                self.fixed_input.resize_(input[0].size()).copy_(input[0])

    ##

    def get_errors(self):
        """ Get netD and netG errors.

        Returns:
            [OrderedDict]: Dictionary containing errors.
        """
        errors = OrderedDict([
            ('err_d', self.err_d.item()),
            ('err_g', self.err_g.item()),
            ('err_g_adv', self.err_g_adv.item()),
            ('err_g_con', self.err_g_con.item()),
            ('err_g_lat', self.err_g_lat.item()),
            ('err_g_kd', self.err_g_kd.item()),
            ('err_d_kd', self.err_d_kd.item())])

        return errors

    ##
    def reinit_d(self):
        """ Initialize the weights of netD
        """
        self.netd.apply(weights_init)
        print('Reloading d net')

    ##
    def get_current_images(self):
        """ Returns current images.

        Returns:
            [reals, fakes, fixed]
        """

        reals = self.input.data
        fakes = self.fake.data
        fixed = self.netg(self.fixed_input)[0].data

        return reals, fakes, fixed

    ##
    def save_weights(self, epoch:int, is_best:bool=False, name:str=None):
        """Save netG and netD weights for the current epoch.

        Args:
            epoch ([int]): Current epoch number.
        """

        weight_dir = os.path.join(
            self.opt.outf, self.opt.name, 'train', 'weights')
        if not os.path.exists(weight_dir):
            os.makedirs(weight_dir)

        if is_best:
            torch.save({'epoch': epoch, 'state_dict': self.netg.state_dict()}, f'{weight_dir}/{name}_netG_best.pth')
            torch.save({'epoch': epoch, 'state_dict': self.netd.state_dict()}, f'{weight_dir}/{name}_netD_best.pth')
        else:
            torch.save({'epoch': epoch, 'state_dict': self.netd.state_dict()}, f"{weight_dir}/{name}_netD_ep{epoch}.pth")
            torch.save({'epoch': epoch, 'state_dict': self.netg.state_dict()}, f"{weight_dir}/{name}_netG_ep{epoch}.pth")

    def load_weights(self, epoch=None, is_best:bool=False, path=None):
        """ Load pre-trained weights of NetG and NetD

        Keyword Arguments:
            epoch {int}     -- Epoch to be loaded  (default: {None})
            is_best {bool}  -- Load the best epoch (default: {False})
            path {str}      -- Path to weight file (default: {None})

        Raises:
            Exception -- [description]
            IOError -- [description]
        """

        if epoch is None and is_best is False:
            raise Exception('Please provide epoch to be loaded or choose the best epoch.')

        if is_best:
            fname_g = f"kd_netG_best.pth"
            fname_d = f"kd_netD_best.pth"
        else:
            fname_g = f"kd_netG_ep{epoch}.pth"
            fname_d = f"kd_netD_ep{epoch}.pth"

        if path is None:
            path_g = f"./output/{self.opt.dataset}/train/weights/{fname_g}"
            path_d = f"./output/{self.opt.dataset}/train/weights/{fname_d}"
        else:
            path_g = f"{path}/{fname_g}"
            path_d = f"{path}/{fname_d}"
            
        # Load the weights of netg and netd.
        print('>> Loading weights...')
        weights_g = torch.load(path_g)['state_dict']
        weights_d = torch.load(path_d)['state_dict']
        try:
            self.netg.load_state_dict(weights_g)
            self.netd.load_state_dict(weights_d)
        except IOError:
            raise IOError("netG weights not found")
        print('   Done.')

    ##
    def update_learning_rate(self):
        """ Update learning rate based on the rule provided in options.
        """
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('   LR = %.7f' % lr)        


    ##
    def train_one_epoch(self, model_teacher):
        """ Train the model for one epoch.
        """

        self.netg.train()
        self.netd.train()

        epoch_iter = 0
        for data in tqdm(self.data.train, leave=False, total=len(self.data.train)):
            self.total_steps += self.opt.batchsize
            epoch_iter += self.opt.batchsize

            self.set_input(data)
            model_teacher.set_input(data)

            teacher_fake = model_teacher.netg.forward(model_teacher.input)
            _, teacher_feat_fake = model_teacher.netd.forward(teacher_fake)

            self.optimize_params(teacher_feat_fake, teacher_fake)

            if self.total_steps % self.opt.print_freq == 0:
                errors = self.get_errors()
                if self.opt.display:
                    counter_ratio = float(epoch_iter) / len(self.data.train.dataset)
                    self.visualizer.plot_current_errors(epoch, counter_ratio, errors)

            if self.total_steps % self.opt.save_image_freq == 0:
                reals, fakes, fixed = self.get_current_images()
                self.visualizer.save_current_images(self.epoch, reals, fakes, fixed)
                if self.opt.display:
                    self.visualizer.display_current_images(reals, fakes, fixed)

        print(f">> Training Epoch {self.epoch+1}/{self.opt.niter}")

    ##
    def train(self, model_teacher=None):
        """ Train the model

        Args:
            model_teacher ([type]): loaded teacher model
        """

        ##
        # TRAIN
        self.total_steps = 0
        best_auc = 0

        # Train for niter epochs.
        print(f">> Training on {self.opt.dataset}")
        for self.epoch in range(self.opt.iter, self.opt.niter):
            self.train_one_epoch(model_teacher)
            res = self.test(model_teacher)
            if res['AUC'] > best_auc:
                best_auc = res['AUC']
                best_ep = res['Epoch']
                self.save_weights(self.epoch, name='kd')
            self.visualizer.print_current_performance(res, best_auc)
        print(f">> Training model on {self.opt.dataset}.[Done]")

        #save best model
        self.save_weights(best_ep, is_best=True, name='kd')
        print(f"Best KD: {best_auc}, Epoch: {best_ep}")

    ##
    def test(self, model_teacher=None):
        """ Test DeepDisaster model.

        Args:
            model_teacher ([type]): loaded teacher model

        Raises:
            IOError: Model weights not found.
        """
        similarity_loss = torch.nn.CosineSimilarity()
        criterion = torch.nn.MSELoss()

        self.netg.eval()
        self.netd.eval()

        for m in self.netg.modules():
            for child in m.children():
                if type(child) == nn.BatchNorm2d:
                    child.track_running_stats = False

        for m in self.netd.modules():
            for child in m.children():
                if type(child) == nn.BatchNorm2d:
                    child.track_running_stats = False

        with torch.no_grad():

            if self.opt.load_weights:
                self.load_weights(is_best=True)

            self.times = []
            self.opt.phase = 'test'
            self.total_steps = 0

            epoch_iter = 0

            # Create big error tensor for the test set.
            self.test_scores = torch.zeros(size=(len(self.data.valid.dataset),), dtype=torch.float32, device=self.device)
            self.gt_labels = torch.zeros(size=(len(self.data.valid.dataset),), dtype=torch.long,    device=self.device)

            print(f"Testing model on {self.opt.dataset}")

            for i, data in enumerate(self.data.valid, 0):
                self.total_steps += self.opt.batchsize
                epoch_iter += self.opt.batchsize

                time_i = time.time()

                Y = data[1]

                model_teacher.set_input(data)
                teacher_fake = model_teacher.netg.forward(model_teacher.input)
                _, teacher_feat_real = model_teacher.netd.forward(model_teacher.input)
                _, teacher_feat_fake = model_teacher.netd.forward(teacher_fake)


                self.set_input(data)
                student_fake = self.netg(self.input)
                _, student_feat_real = self.netd(self.input)
                _, student_feat_fake = self.netd(student_fake)


               # Calculate the score.
                si = self.input.size()
                sz = student_feat_real.size()
                rec = (self.input - student_fake).view(si[0], si[1] * si[2] * si[3])
                lat = (student_feat_real - student_feat_fake).view(sz[0], sz[1] * sz[2] * sz[3])
                rec = torch.mean(torch.pow(rec, 2), dim=1)
                lat = torch.mean(torch.pow(lat, 2), dim=1)
                score = 0.4*rec + 0.2*lat
            
        
                abs_loss_f = torch.mean((student_feat_fake - teacher_feat_fake) ** 2, dim=(1, 2, 3))
                loss_f = 1 - similarity_loss(student_feat_fake.view(student_feat_fake.shape[0], -1),
                             teacher_feat_fake.view(teacher_feat_fake.shape[0], -1))
                
                abs_loss_x_hat = torch.mean((student_fake- teacher_fake) ** 2, dim=(1, 2, 3))
                loss_x_hat = 1 - similarity_loss(student_fake.view(student_fake.shape[0], -1),
                             teacher_fake.view(teacher_fake.shape[0], -1))


                total_loss = loss_f + loss_x_hat + self.opt.lambdaa * (abs_loss_x_hat + abs_loss_f)

                score += 0.4*total_loss
                time_o = time.time()
                

                self.test_scores[i*self.opt.batchsize: i*self.opt.batchsize + score.size(0)] = score.reshape(score.size(0))
                self.gt_labels[i*self.opt.batchsize: i*self.opt.batchsize + score.size(0)] = self.gt.reshape(score.size(0))

            self.times.append(time_o - time_i)
            # Save test images.
            if self.opt.save_test_images:
                dst = os.path.join(self.opt.outf, 'test', 'images')
                if not os.path.isdir(dst):
                    os.makedirs(dst)
                    real, fake, _ = self.get_current_images()
                    vutils.save_image(real, '%s/real_%03d.eps' % (dst, i+1), normalize=True)
                    vutils.save_image(fake, '%s/fake_%03d.eps' % (dst, i+1), normalize=True)

            # Measure inference time.
            self.times = np.array(self.times)
            self.times = np.mean(self.times[:100] * 1000)


            label_score = list(zip(self.gt_labels.cpu().data.numpy().tolist(), self.test_scores.cpu().data.numpy().tolist()))
        
            labels, scores = zip(*label_score)
            labels = np.array(labels)
            scores = np.array(scores)

            fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
            roc_auc = auc(fpr, tpr)
            roc_auc = round(roc_auc, 4)
            performance = OrderedDict([('KD LOSS: Avg Run Time (ms/batch)', self.times), ('AUC', roc_auc), ('Epoch', self.epoch+1)])

        
            # Scale error vector between [0, 1]
            # self.test_scores = (self.test_scores - torch.min(self.test_scores)) / (torch.max(self.test_scores) - torch.min(self.an_scores))
            # auc = evaluate(self.gt_labels, self.test_scores, metric=self.opt.metric)
            # performance = OrderedDict([('Avg Run Time (ms/batch)', self.times), ('AUC', auc)])

            if self.opt.display_id > 0 and self.opt.phase == 'test':
                counter_ratio = float(epoch_iter) / len(self.data.valid.dataset)
                self.visualizer.plot_performance(self.epoch, counter_ratio, performance)

            return performance

        ##
  
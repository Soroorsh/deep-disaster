import torch
import time
import numpy as np

from torch import nn
from tqdm import tqdm
from copy import deepcopy
from collections import OrderedDict
from torch.autograd import Variable
from sklearn.metrics import roc_curve, auc
from scipy.ndimage.filters import gaussian_filter

from .utils import *

def detection_test(teacher, student, opt):
    """ Test DeepDisaster model .

        Args:
            teacher ([type]): loaded teacher model
            student ([type]): loaded student model

        Raises:
                IOError: Model weights not found.

        Returns:
            [OrderedDict]: calculated AUC-ROC performance
    """    
    similarity_loss = torch.nn.CosineSimilarity()
    criterion = torch.nn.MSELoss()
    label_score = []

    student.netg.eval()
    student.netd.eval()
    
    for m in student.netg.modules():
        for child in m.children():
            if type(child) == nn.BatchNorm2d:
                child.track_running_stats = False

  
    for m in student.netd.modules():
        for child in m.children():
            if type(child) == nn.BatchNorm2d:
                child.track_running_stats = False

    with torch.no_grad():

        times = []
        opt.phase = 'test'

        test_scores = torch.zeros(size=(len(student.data.valid.dataset),), dtype=torch.float32, device=student.device)
        gt_labels = torch.zeros(size=(len(student.data.valid.dataset),), dtype=torch.long, device=student.device)


        print(f"Testing model on {opt.dataset}")

        for i, data in enumerate(student.data.valid, 0):
            Y = data[1]
            images = data[0]
        
            sample_fname = student.data.valid.dataset.samples[i*opt.batchsize : (i+1)*opt.batchsize]
           
            time_i = time.time()

            # Forward - Pass
            teacher.set_input(data)
            teacher_fake = teacher.netg.forward(teacher.input)

            _, teacher_feat_real = teacher.netd.forward(teacher.input)
            _, teacher_feat_fake = teacher.netd.forward(teacher_fake)


            student.set_input(data)
            student_fake = student.netg(student.input)

            _, student_feat_real = student.netd(student.input)
            _, student_feat_fake = student.netd(student_fake)

            # Calculate the score.
            si = student.input.size()
            sz = student_feat_real.size()
            rec = (student.input - student_fake).view(si[0], si[1] * si[2] * si[3])
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

            total_loss = loss_f + loss_x_hat + opt.lambdaa * (abs_loss_x_hat + abs_loss_f)
              
            score += 0.4*total_loss
            time_o = time.time()
         
          
            test_scores[i*opt.batchsize: i*opt.batchsize + score.size(0)] = score.reshape(score.size(0))
            gt_labels[i*opt.batchsize: i*opt.batchsize + score.size(0)] = student.gt.reshape(score.size(0))
            
            times.append(time_o - time_i)


        # Measure inference time.
        times = np.array(times)
        times = np.mean(times[:100] * 1000)

        if opt.save_test_images:
            dst = os.path.join(opt.outf, 'test', 'images')
            if not os.path.isdir(dst):
                os.makedirs(dst)
                real, fake, _ = student.get_current_images()
                vutils.save_image(real, '%s/real_%03d.eps' % (dst, i+1), normalize=True)
                vutils.save_image(fake, '%s/fake_%03d.eps' % (dst, i+1), normalize=True)


        label_score = list(zip(gt_labels.cpu().data.numpy().tolist(), test_scores.cpu().data.numpy().tolist()))
    
        labels, scores = zip(*label_score)
        labels = np.array(labels)
        scores = np.array(scores)

        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
        roc_auc = auc(fpr, tpr)
        roc_auc = round(roc_auc, 4)
        performance = OrderedDict([('KD LOSS: Avg Run Time (ms/batch)', times), ('AUC', roc_auc)])

        
        if student.opt.display_id > 0 and student.opt.phase == 'test':
            counter_ratio = float(epoch_iter) / len(student.data.valid.dataset)
            student.visualizer.plot_performance(epoch, counter_ratio, performance)
        
        return performance



def localization_test(teacher, student, config):
    """ Run localization test  on DeepDisaster model according to the localization_method parameter.

        Args:
            teacher ([type]): loaded teacher model
            student ([type]): loaded student model
            config ([type]) : config

        Returns:
            [np.array]: ouput gradients
    """    

    localization_method = config.localization_method
    if localization_method == 'gradients':
        grad = gradients_localization(student, teacher, config)
    if localization_method == 'smooth_grad':
        grad = smooth_grad_localization(student, teacher, config)
    if localization_method == 'gbp':
        grad = gbp_localization(student, teacher, config)

    return grad 




def gradients_localization(student, teacher, config):
    """ Calculates vanilla backpropagation.

        Args:

            teacher ([type]): loaded teacher model
            student ([type]): loaded student model
            config ([type]) : config

        Returns:
            [np.array]: ouput gradients
    """    
    student.netg.eval()

    for m in student.netg.modules():
        for child in m.children():
            if type(child) == nn.BatchNorm2d:
                child.track_running_stats = False

    print("Vanilla Backpropagation:")
    res = []
    for i, data in enumerate(student.data.valid, 0):
        grad = grad_calc(data, student, teacher, config)
        temp = np.zeros((grad.shape[0], grad.shape[2], grad.shape[3]))
        for i in range(grad.shape[0]):
            grad_temp = convert_to_grayscale(grad[i].cpu().numpy())
            grad_temp = grad_temp.squeeze(0)
            grad_temp = gaussian_filter(grad_temp, sigma=4)
            temp[i] = grad_temp
        res.extend(temp)
    return np.array(res)


def gbp_localization(student, teacher, config):
    """ Calculates guided backpropagation method.

        Args:

            teacher ([type]): loaded teacher model
            student ([type]): loaded student model
            config ([type]) : config

        Returns:
            [np.array]: ouput gradients
    """    
    student.netg.eval()
    for m in student.netg.modules():
        for child in m.children():
            if type(child) == nn.BatchNorm2d:
                child.track_running_stats = False

    print("GBP Method:")

    grad1 = None
    grad_all = []
    cnt = 0
    grad1 = np.zeros((len(student.data.valid.dataset), 1, 64, 64), dtype=np.float32)
    for i, data in enumerate(student.data.valid, 0):
        X, Y = data
        for j, x in enumerate(X,0):
            data = x.view(1, 3, 64, 64)

            GBP = GuidedBackprop(student, teacher, 'cuda:0')
            gbp_saliency = abs(GBP.generate_gradients(data.squeeze(0), Y[j], config))
            gbp_saliency = (gbp_saliency - min(gbp_saliency.flatten())) / (
                    max(gbp_saliency.flatten()) - min(gbp_saliency.flatten()))
            saliency = gbp_saliency

            saliency = gaussian_filter(saliency, sigma=4)
            grad1[cnt] = saliency
            cnt += 1

    grad1 = grad1.reshape(-1, 64, 64)
    return grad1


def smooth_grad_localization(student, teacher, config):
    """ Calculates Smooth gradient method.

    Args:

        teacher ([type]): loaded teacher model
        student ([type]): loaded student model
        config ([type]) : config

    Returns:
        [np.array]: ouput gradients
    """    
    student.netg.eval()

    for m in student.netg.modules():
            for child in m.children():
                if type(child) == nn.BatchNorm2d:
                    child.track_running_stats = False

    print("Smooth Grad Method:")
    grad1 = None
    grad_all = []
    cnt = 0
    grad1 = np.zeros((len(student.data.valid.dataset), 1, 64, 64), dtype=np.float32)
    for i, data in enumerate(student.data.valid, 0):
        X, Y = data
        for j, x in enumerate(X,0):
            data = x.view(1, 3, 64, 64)

            vbp = VanillaSaliency(student, teacher, 'cuda:0', config)

            smooth_grad_saliency = abs(generate_smooth_grad(data.squeeze(0),Y[j], 50, 0.05, vbp))
            smooth_grad_saliency = (smooth_grad_saliency - min(smooth_grad_saliency.flatten())) / (
                    max(smooth_grad_saliency.flatten()) - min(smooth_grad_saliency.flatten()))
            saliency = smooth_grad_saliency

            saliency = gaussian_filter(saliency, sigma=4)
            grad1[cnt] = saliency
            cnt += 1
        
    grad1 = grad1.reshape(-1, 64, 64)
    return grad1


def grad_calc(inputs, student, teacher, config):
    
    """ Calculates gradients based on the loss.

        Args:

            inputs ([type]): a list of input-images and their corresponding labels
            teacher ([type]): loaded teacher model
            student ([type]): loaded student model
            config ([type]) : config

        Returns:
            [np.array]: ouput gradients
    """    
    X = inputs[0]
    Y= inputs[1]
    X = X.cuda()
    X.requires_grad = True

    temp = torch.zeros(X.shape)
    lamda = config.lambdaa
    criterion = nn.MSELoss()
    similarity_loss = torch.nn.CosineSimilarity()

    for i in range(X.shape[0]):

        student.set_input([X[i].unsqueeze(0),Y[i]])
        teacher.set_input([X[i].unsqueeze(0),Y[i]])
  
        teacher.fake = teacher.netg(X[i].unsqueeze(0))
        student.fake = student.netg(X[i].unsqueeze(0))

        _, teacher.feat_real = teacher.netd.forward(teacher.input)
        _, teacher.feat_fake = teacher.netd.forward(teacher.fake)

        student.pred_real, student.feat_real = student.netd(student.input)
        student.pred_fake, student.feat_fake = student.netd(student.fake)
   
        si = student.input.size()
        sz = student.feat_real.size()
        rec = (student.input - student.fake).view(si[0], si[1] * si[2] * si[3])
        lat = (student.feat_real - student.feat_fake).view(sz[0], sz[1] * sz[2] * sz[3])
        rec = torch.mean(torch.pow(rec, 2), dim=1)
        lat = torch.mean(torch.pow(lat, 2), dim=1)
    

        abs_loss = criterion(student.fake, teacher.fake)
        loss = torch.mean(1 - similarity_loss(student.fake.view(student.fake.shape[0], -1),
                                 teacher.fake.view(teacher.fake.shape[0], -1)))

        total_loss =  0.4*rec + 0.2*lat + 0.4*(loss+ lamda * abs_loss)

        student.netg.zero_grad()
        student.netd.zero_grad()
        total_loss.backward()
       
        temp[i] = X.grad[i]

    return temp


class VanillaSaliency():

    """ VanillaSaliency Class.
    """    

    def __init__(self, student, teacher, device, config):
        self.student = student
        self.teacher = teacher
        self.device = device
        self.config = config

        self.student.netg.eval()
 
        for m in self.student.netg.modules():
            for child in m.children():
                if type(child) == nn.BatchNorm2d:
                    child.track_running_stats = False

    def generate_saliency(self, data, label, make_single_channel=True):
        """ Generate Sailency map based on loss.

        Args:

            data ([type]): input images
            label ([type]): corresponding input labels
            make_single_channel ([type]) : defines convert grads to grayscale or not

        Returns:
            [np.array]: ouput gradients
        """    
        data_var_sal = Variable(data).to(self.device)
        self.student.netg.zero_grad()
        if data_var_sal.grad is not None:
            data_var_sal.grad.data.zero_()
        data_var_sal.requires_grad_(True)

        lamda = self.config.lambdaa
        criterion = nn.MSELoss()
        similarity_loss = torch.nn.CosineSimilarity()

        self.student.set_input([data_var_sal.unsqueeze(0),label])
        self.teacher.set_input([data_var_sal.unsqueeze(0),label])
  

        teacher_fake = self.teacher.netg.forward(data_var_sal.unsqueeze(0))
        student_fake = self.student.netg(data_var_sal.unsqueeze(0))

        _, teacher_feat_real = self.teacher.netd.forward(self.teacher.input)
        _, teacher_feat_fake = self.teacher.netd.forward(teacher_fake)

        _, student_feat_real = self.student.netd(self.student.input)
        _, student_feat_fake = self.student.netd(student_fake)


        si = self.student.input.size()
        sz = student_feat_real.size()
        rec = (self.student.input - student_fake).view(si[0], si[1] * si[2] * si[3])
        lat = (student_feat_real - student_feat_fake).view(sz[0], sz[1] * sz[2] * sz[3])
        rec = torch.mean(torch.pow(rec, 2), dim=1)
        lat = torch.mean(torch.pow(lat, 2), dim=1)


        abs_loss = criterion(student_fake, teacher_fake)
        loss = torch.mean(1 - similarity_loss(student_fake.view(student_fake.shape[0], -1),
                                 teacher_fake.view(teacher_fake.shape[0], -1)))

        total_loss =  0.4*rec + 0.2*lat + 0.4*(loss+ lamda * abs_loss)        

        self.student.netg.zero_grad()
        self.student.netd.zero_grad()

        total_loss.backward()
        grad = data_var_sal.grad.data.detach().cpu()

        if make_single_channel:
            grad = np.asarray(grad.detach().cpu().squeeze(0))
            grad = convert_to_grayscale(grad)
        else:
            grad = np.asarray(grad)
        return grad


def generate_smooth_grad(data,label, param_n, param_sigma_multiplier, vbp, single_channel=True):
    """ Generate Smooth gradients  based on loss. 
        by adding gussian noise to the vanilla gradients approch

    Args:

        data ([type]): input images
        label ([type]): corresponding input labels
        param_n (int): Amount of images used to smooth gradient
        param_sigma_multiplier (int): Sigma multiplier when calculating std of noise
        vbp (class): VanillaSaliency class
        single_channel ([type]) : defines convert grads to grayscale or not

    Returns:
        [np.array]: ouput gradients
    """       
    smooth_grad = None

    mean = 0
    sigma = param_sigma_multiplier / (torch.max(data) - torch.min(data)).item()
    VBP = vbp
    for x in range(param_n):
        noise = Variable(data.data.new(data.size()).normal_(mean, sigma ** 2))
        noisy_img = data + noise
        vanilla_grads = VBP.generate_saliency(noisy_img, label, single_channel)
        if not isinstance(vanilla_grads, np.ndarray):
            vanilla_grads = vanilla_grads.detach().cpu().numpy()
        if smooth_grad is None:
            smooth_grad = vanilla_grads
        else:
            smooth_grad = smooth_grad + vanilla_grads

    smooth_grad = smooth_grad / param_n
    return smooth_grad

class GuidedBackprop():
    """
    Produces gradients generated using guided back propagation
    """
    def __init__(self, student, teacher, device):
        self.student = student
        self.teacher = teacher
        self.gradients = None
        self.forward_relu_outputs = []
        self.device = device
        self.hooks = []
        self.update_relus()

     

    def update_relus(self):

        """
            Updates relu activation functions:
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """

        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for module in self.student.netg.modules():
            if isinstance(module, nn.ReLU):
                self.hooks.append(module.register_backward_hook(relu_backward_hook_function))
                self.hooks.append(module.register_forward_hook(relu_forward_hook_function))

    def generate_gradients(self, input_image, label, config, make_single_channel=True):
        """
        generates the gradients

        Args:

            input_image ([type]): input images
            label ([type]): corresponding input labels
            config ([type]) : config
            make_single_channel ([type]) : defines convert grads to grayscale or not

        Returns:
            [np.array]: ouput gradients
        """    
        vanillaSaliency = VanillaSaliency(self.student, self.teacher, self.device, config=config)
        sal = vanillaSaliency.generate_saliency(input_image, label , make_single_channel)
        if not isinstance(sal, np.ndarray):
            sal = sal.detach().cpu().numpy()
        for hook in self.hooks:
            hook.remove()
        return sal



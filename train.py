import sys
import time 
import torch
import pickle
import numpy as np

from tqdm import tqdm

from options import Options
from lib.model.model import DeepDisaster
from lib.data.dataloader import load_data



def main():
    """
    Train Funtion: first loads train and test data, also loads the models,
    then start training the student network with the help of the pre-trained teacher. 
    
    """
 
    opt = Options().parse()
    data = load_data(opt)  # return Data(train_dl, valid_dl)
    
    model_teacher = DeepDisaster(opt, data, teacher=True)
    model_student = DeepDisaster(opt, data, teacher=False)

    """ Train the model """
    model_student.train(model_teacher)


if __name__ == '__main__':
    main()
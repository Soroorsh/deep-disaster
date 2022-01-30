import os
import time 
import torch
import numpy as np

from tqdm import tqdm

from options import Options
from lib.model.model import DeepDisaster
from lib.utils.visualizer import Visualizer
from lib.data.dataloader import load_test_data
from lib.utils.test_functions import localization_test, detection_test



def main():
    """
    Test Funtion: first loads test data and the models,
    then performs localization or detection test according to the localization_test argument

        """

    opt = Options().parse()
    data = load_test_data(opt)  # return Data(valid_dl)

    model_teacher = DeepDisaster(opt, data, teacher=True)
    model_student = DeepDisaster(opt, data, teacher=False)

    if opt.localization_test: 
        
        print(f'Calculating localization maps ...')
        grad = localization_test(model_teacher, model_student, opt)
        visualizer =Visualizer(opt)
        visualizer.save_localization_images(model_student,grad)
        print(f'Done!')


    else:
        performance= detection_test(model_teacher, model_student, opt)
        print(f'test perfomance: {performance}')

if __name__ == '__main__':
    main()

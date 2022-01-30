"""
LOAD DATA from file.
"""

# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


class Data:
    """ Dataloader containing train and valid sets.
    """
    def __init__(self, train, valid):
        self.train = train
        self.valid = valid


def load_test_data(opt):
    """ Load Data

    Args:
        opt ([type]): Argument Parser

    Raises:
        IOError: Cannot Load Dataset

    Returns:
        [type]: dataloader
    """

    ##
    # LOAD DATA SET
    if opt.dataroot == '':
        opt.dataroot = './data/DAD/'

    # FOLDER

    transform = transforms.Compose([transforms.Resize(opt.isize),
                                    transforms.CenterCrop(opt.isize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

    test_ds = ImageFolder(os.path.join(opt.dataroot, opt.dataset, 'test'), transform)

    ## DATALOADER
    test_dl = DataLoader(dataset=test_ds, batch_size=opt.batchsize, shuffle=False, drop_last=False)

    return Data(None, test_dl)


def load_data(opt):
    """ Load Data

    Args:
        opt ([type]): Argument Parser

    Raises:
        IOError: Cannot Load Dataset

    Returns:
        [type]: dataloader
    """

    ##
    # LOAD DATA SET
    if opt.dataroot == '':
        opt.dataroot = './data/DAD/'

    # LOAD DATA FROM FOLDER

    transform = transforms.Compose([transforms.Resize(opt.isize),
                                    transforms.CenterCrop(opt.isize),
                                    transforms.RandomHorizontalFlip(50),
                                    transforms.RandomVerticalFlip(50),
                                    transforms.RandomRotation(degrees =30),
                                    transforms.RandomCrop(64, padding = 4),
                                    transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0, hue=0),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

    train_ds = ImageFolder(os.path.join(opt.dataroot, opt.dataset, 'train'), transform)
    valid_ds = ImageFolder(os.path.join(opt.dataroot, opt.dataset, 'test'), transform)

    ## DATALOADER
    train_dl = DataLoader(dataset=train_ds, batch_size=opt.batchsize, shuffle=True, drop_last=True)
    valid_dl = DataLoader(dataset=valid_ds, batch_size=opt.batchsize, shuffle=False, drop_last=False)

    return Data(train_dl, valid_dl)


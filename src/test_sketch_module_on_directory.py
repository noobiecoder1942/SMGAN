import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import cv2
import scipy.ndimage as pyimg
import random
import os
import sys

import torch
from models import SketchModule
from utils import load_image, to_data, to_var, visualize, save_image, gaussian, weights_init
# from utils import load_train_batchfnames, prepare_text_batch

# from utils import 

import time
import argparse
import os
import shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

opts = argparse.ArgumentParser()
opts.GB_nlayers = 6
opts.DB_nlayers = 5
opts.GB_nf = 32
opts.DB_nf = 32
opts.gpu = True
opts.epochs = 6
opts.save_GB_name = '../save/GB.ckpt'
opts.batchsize = 64
opts.text_path = '../data/rawtext/yaheiB/train'
opts.augment_text_path = '../data/rawtext/augment'
opts.text_datasize = 1760
opts.augment_text_datasize = 5
opts.Btraining_num = 12800


testing_directory = '/home/abhirag/cv_project/experiment/data/test_logos_processed'
destination_directory = '/home/abhirag/cv_project/experiment/data/test_logos_output_exp3_epoch5'

def custom_load_train_batchfnames(text_path, augment_path, batch_size, usenum_text, usenum_augment, trainnum):
    fnames = [os.path.join(text_path, i) for i in os.listdir(text_path)]

    new_fnames = []
    
    for i in range(trainnum):
        new_fnames.append(fnames[i % len(fnames)])

    random.shuffle(new_fnames)
    trainbatches = [new_fnames[x:x+batch_size] for x in range(0, len(new_fnames), batch_size)]
    fnames2 = [os.path.join(augment_path, i) for i in os.listdir(augment_path)]
    new_fnames2 = []
    
    for i in range(trainnum):
        new_fnames2.append(fnames2[i % len(fnames2)])

    random.shuffle(new_fnames2)
    augmentbatches = [new_fnames2[x:x+batch_size] for x in range(0, len(new_fnames2), batch_size)]
    return trainbatches, augmentbatches


pil2tensor = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5,0.5,0.5])])
tensor2pil = transforms.ToPILImage()

def prepare_text_batch(batchfnames, wd=256, ht=256, anglejitter=False):
    img_list = []
    for fname in batchfnames:
        img = Image.open(fname).convert('RGB')
        ori_wd, ori_ht = img.size
#         print(fname)
        # check if image dimensions are greater than 256*256 otherwise cropping will throw error
        if ori_ht > 256 and ori_wd > 256:
            w = random.randint(0,ori_wd-wd)
            h = random.randint(0,ori_ht-ht)
            img = img.crop((w,h,w+wd,h+ht))
        else:
            img = img.resize((256, 256))
        if anglejitter:
            random_angle = 90 * random.randint(0,3)
            img = img.rotate(random_angle)
        img = pil2tensor(img)         
        img = img.unsqueeze(dim=0)
        img_list.append(img)
    return torch.cat(img_list, dim=0)




if __name__ == "__main__":
    if(len(sys.argv) != 4):
        raise Exception('Please provide 3 commandline arguments. ckpt testing_dir destination_dir')


    testing_directory = sys.argv[2] 
    destination_directory = sys.argv[3] 
    ckpt_file = sys.argv[1]


    if os.path.exists(destination_directory):
        print('REMOVING DIRECTORY')
        shutil.rmtree(destination_directory)

    print('CREATING DIRECTORY')
    os.mkdir(destination_directory)

    netSketch = SketchModule(opts.GB_nlayers, opts.DB_nlayers, opts.GB_nf, opts.DB_nf, opts.gpu)
    if opts.gpu:
        netSketch.cuda()
    netSketch.init_networks(weights_init)
    netSketch.train()

    netSketch.load_state_dict(torch.load(ckpt_file))


    netSketch.eval()

    list_of_files = [os.path.join(testing_directory, i) for i in os.listdir(testing_directory)]

    for file in list_of_files:
        print('PROCESSING == ', file)
        I = load_image(file)
        I = to_var(I[:,:,:,0:I.size(3)])
        result = netSketch(I, -1.)

        Image.fromarray(((to_data(result[0]).numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)).save(os.path.join(destination_directory, file.split('/')[-1]))


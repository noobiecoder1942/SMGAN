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
from models import SketchModule
from utils import load_image, to_data, to_var, visualize, save_image, gaussian, weights_init
# from utils import load_train_batchfnames, prepare_text_batch

from utils import custom_load_train_batchfnames, prepare_text_batch

import time
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

opts = argparse.ArgumentParser()
opts.GB_nlayers = 8
opts.DB_nlayers = 5
opts.GB_nf = 128
opts.DB_nf = 64
opts.gpu = True
opts.epochs = 6
opts.save_GB_name = '../save/GB_exp15.ckpt'
opts.batchsize = 32
opts.text_path = '../data/rawtext/yaheiB/train'
opts.augment_text_path = '../data/new_augment'
opts.text_datasize = 1760
opts.augment_text_datasize = 22
opts.Btraining_num = 25600

ep_320 = 2
ep_256 = 1
ep_192 = 0
ep_128 = 3
ep_64 = 1
ep_32 = 2

# fnames, fnames2 = custom_load_train_batchfnames(opts.text_path, opts.augment_text_path, opts.batchsize, opts.text_datasize, opts.augment_text_datasize, trainnum = opts.Btraining_num)

pil2tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5,0.5,0.5])])
tensor2pil = transforms.ToPILImage()

print('--- create model ---')
netSketch = SketchModule(opts.GB_nlayers, opts.DB_nlayers, opts.GB_nf, opts.DB_nf, opts.gpu)
if opts.gpu:
    netSketch.cuda()
netSketch.init_networks(weights_init)
netSketch.train()

print('--- training ---')

##########################################################################################################

for epoch in range(ep_320):

    opts.batchsize = 16
    opts.Btraining_num = 12800

    curr_time = time.time()
    itr = 0
    fnames, fnames2 = custom_load_train_batchfnames(opts.text_path, opts.augment_text_path, opts.batchsize, opts.text_datasize, opts.augment_text_datasize, trainnum = opts.Btraining_num)

    
    for ii in range(len(fnames)):
        fnames[ii][0:opts.batchsize//8-1] = fnames2[ii][0:opts.batchsize//8-1]
    
    for fname in fnames:
        random.shuffle(fname)
        itr += 1
        t = prepare_text_batch(fname, 320, 320, anglejitter=True)
        t = to_var(t) if opts.gpu else t
        losses = netSketch.one_pass(t, [l/4.-1. for l in range(0,9)])
        print('Epoch [%d/%d][%03d/%03d]' %(epoch+1, opts.epochs, itr, len(fnames)), end=': ')
        print('LDadv: %+.3f, LGadv: %+.3f, Lrec: %+.3f'%(losses[0], losses[1], losses[2]))

    end_time = time.time()
    print('Epoch = ', epoch+1, 'time taken = ', str(end_time - curr_time))

##########################################################################################################

    
for epoch in range(ep_256):

    opts.batchsize = 32
    opts.Btraining_num = 12800

    curr_time = time.time()
    itr = 0
    fnames, fnames2 = custom_load_train_batchfnames(opts.text_path, opts.augment_text_path, opts.batchsize, opts.text_datasize, opts.augment_text_datasize, trainnum = opts.Btraining_num)

    
    for ii in range(len(fnames)):
        fnames[ii][0:opts.batchsize//8-1] = fnames2[ii][0:opts.batchsize//8-1]
    
    for fname in fnames:
        random.shuffle(fname)
        itr += 1
        t = prepare_text_batch(fname, 256, 256, anglejitter=True)
        t = to_var(t) if opts.gpu else t
        losses = netSketch.one_pass(t, [l/4.-1. for l in range(0,9)])
        print('Epoch [%d/%d][%03d/%03d]' %(epoch+1, opts.epochs, itr, len(fnames)), end=': ')
        print('LDadv: %+.3f, LGadv: %+.3f, Lrec: %+.3f'%(losses[0], losses[1], losses[2]))

    end_time = time.time()
    print('Epoch = ', epoch+1, 'time taken = ', str(end_time - curr_time))

##########################################################################################################


    
for epoch in range(ep_192):

    opts.batchsize = 32
    opts.Btraining_num = 12800

    curr_time = time.time()
    itr = 0
    fnames, fnames2 = custom_load_train_batchfnames(opts.text_path, opts.augment_text_path, opts.batchsize, opts.text_datasize, opts.augment_text_datasize, trainnum = opts.Btraining_num)

    
    for ii in range(len(fnames)):
        fnames[ii][0:opts.batchsize//8-1] = fnames2[ii][0:opts.batchsize//8-1]
    
    for fname in fnames:
        random.shuffle(fname)
        itr += 1
        t = prepare_text_batch(fname, 192, 192, anglejitter=True)
        t = to_var(t) if opts.gpu else t
        losses = netSketch.one_pass(t, [l/4.-1. for l in range(0,9)])
        print('Epoch [%d/%d][%03d/%03d]' %(epoch+1, opts.epochs, itr, len(fnames)), end=': ')
        print('LDadv: %+.3f, LGadv: %+.3f, Lrec: %+.3f'%(losses[0], losses[1], losses[2]))

    end_time = time.time()
    print('Epoch = ', epoch+1, 'time taken = ', str(end_time - curr_time))
    

##########################################################################################################

    
for epoch in range(ep_128):

    opts.batchsize = 64
    opts.Btraining_num = 25600

    curr_time = time.time()
    itr = 0
    fnames, fnames2 = custom_load_train_batchfnames(opts.text_path, opts.augment_text_path, opts.batchsize, opts.text_datasize, opts.augment_text_datasize, trainnum = opts.Btraining_num)
    
    for ii in range(len(fnames)):
        fnames[ii][0:opts.batchsize//8-1] = fnames2[ii][0:opts.batchsize//8-1]
    
    for fname in fnames:
        random.shuffle(fname)
        itr += 1
        t = prepare_text_batch(fname, 128, 128, anglejitter=True)
        t = to_var(t) if opts.gpu else t
        losses = netSketch.one_pass(t, [l/4.-1. for l in range(0,9)])
        print('Epoch [%d/%d][%03d/%03d]' %(epoch+1, opts.epochs, itr, len(fnames)), end=': ')
        print('LDadv: %+.3f, LGadv: %+.3f, Lrec: %+.3f'%(losses[0], losses[1], losses[2]))

    end_time = time.time()
    print('Epoch = ', epoch+1, 'time taken = ', str(end_time - curr_time))

##########################################################################################################

    
for epoch in range(ep_64):

    opts.batchsize = 64
    opts.Btraining_num = 12800

    curr_time = time.time()
    itr = 0
    fnames, fnames2 = custom_load_train_batchfnames(opts.text_path, opts.augment_text_path, opts.batchsize, opts.text_datasize, opts.augment_text_datasize, trainnum = opts.Btraining_num)
    
    for ii in range(len(fnames)):
        fnames[ii][0:opts.batchsize//8-1] = fnames2[ii][0:opts.batchsize//8-1]
    
    for fname in fnames:
        random.shuffle(fname)
        itr += 1
        t = prepare_text_batch(fname, 64, 64, anglejitter=True)
        t = to_var(t) if opts.gpu else t
        losses = netSketch.one_pass(t, [l/4.-1. for l in range(0,9)])
        print('Epoch [%d/%d][%03d/%03d]' %(epoch+1, opts.epochs, itr, len(fnames)), end=': ')
        print('LDadv: %+.3f, LGadv: %+.3f, Lrec: %+.3f'%(losses[0], losses[1], losses[2]))

    end_time = time.time()
    print('Epoch = ', epoch+1, 'time taken = ', str(end_time - curr_time))

##########################################################################################################

      
for epoch in range(ep_32):


    opts.batchsize = 64
    opts.Btraining_num = 12800

    curr_time = time.time()
    itr = 0
    fnames, fnames2 = custom_load_train_batchfnames(opts.text_path, opts.augment_text_path, opts.batchsize, opts.text_datasize, opts.augment_text_datasize, trainnum = opts.Btraining_num)
    
    for ii in range(len(fnames)):
        fnames[ii][0:opts.batchsize//8-1] = fnames2[ii][0:opts.batchsize//8-1]
    
    for fname in fnames:
        random.shuffle(fname)
        itr += 1
        t = prepare_text_batch(fname, 32, 32, anglejitter=True)
        t = to_var(t) if opts.gpu else t
        losses = netSketch.one_pass(t, [l/4.-1. for l in range(0,9)])
        print('Epoch [%d/%d][%03d/%03d]' %(epoch+1, opts.epochs, itr, len(fnames)), end=': ')
        print('LDadv: %+.3f, LGadv: %+.3f, Lrec: %+.3f'%(losses[0], losses[1], losses[2]))

    end_time = time.time()
    print('Epoch = ', epoch+1, 'time taken = ', str(end_time - curr_time))
        

##########################################################################################################


print('--- save ---')
# directory
torch.save(netSketch.state_dict(), opts.save_GB_name)

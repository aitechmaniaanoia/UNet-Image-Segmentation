import os
from os.path import isdir, exists, abspath, join

import random

import numpy as np
from PIL import Image, ImageOps, ImageEnhance

class DataLoader():
    def __init__(self, root_dir='data/cells', batch_size=2, test_percent=.1):
        self.batch_size = batch_size
        self.test_percent = test_percent

        self.root_dir = abspath(root_dir)
        self.data_dir = join(self.root_dir, 'scans')
        self.labels_dir = join(self.root_dir, 'labels')

        self.files = os.listdir(self.data_dir)

        self.data_files = [join(self.data_dir, f) for f in self.files]
        self.label_files = [join(self.labels_dir, f) for f in self.files]

    def __iter__(self):
        n_train = self.n_train()

        if self.mode == 'train':
            current = 0
            endId = n_train
        elif self.mode == 'test':
            current = n_train
            endId = len(self.data_files)

        while current < endId:
            current += 1

            # todo: load images and labels
            # hint: scale images between 0 and 1
            # hint: if training takes too long or memory overflow, reduce image size!
            
            data_image = Image.open(self.data_files[current-1])
            label_image = Image.open(self.label_files[current-1])
            
            #enh_con = ImageEnhance.Contrast(data_image)
            #contrast = 1.5
            #data_image = enh_con.enhance(contrast)
            data_image = ImageOps.equalize(data_image)     
            
            size = 512, 512
            
            if self.mode == 'train':
            
                data_image.thumbnail(size, Image.ANTIALIAS)
                label_image.thumbnail(size, Image.ANTIALIAS)
            
            ### DATA AUGMENTATION
            aug = np.random.randint(0,2,size = 1)
            
            if aug == 0: # flip
                data_new = ImageOps.mirror(data_image)
                label_new = ImageOps.mirror(label_image)
                
            elif aug == 1: # rotate
                data_new = data_image.rotate(90)
                label_new = label_image.rotate(90)
                
            elif aug == 2: # Zoom
                #w,h = data_image.size
                data_new = data_image.crop((5,5,size[0]-5,size[1]-5))
                label_new = label_image.crop((5,5,size[0]-5,size[1]-5))
                
                data_new = data_new.resize(size)
                label_new = label_new.resize(size)
            
            data_image = np.array(data_image)
            label_image = np.array(label_image)
            
            data_image = data_image / np.max(data_image)
            #data_image = data_image / 255
            
            data_new = np.array(data_new)
            label_new = np.array(label_new)
            
            data_new = data_new / np.max(data_new)
            #data_new = data_new / 255
            
            yield (data_image, label_image)
            yield (data_new, label_new)

    def setMode(self, mode):
        self.mode = mode

    def n_train(self):
        data_length = len(self.data_files)
        return np.int_(data_length - np.floor(data_length * self.test_percent))
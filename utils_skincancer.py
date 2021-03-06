# -*- coding: utf-8 -*-
"""

modul: utils for skin cancer py
modul author: Christoph Doerr

"""

import numpy as np
import shutil
import os.path
import os
from pathlib import Path
from keras.preprocessing.image import ImageDataGenerator

def identify_duplicates(x, metadata):
    """ Identify melanom id with multple images
    Input metadata: pandas data series 
    """
    unique_list = list(metadata)
    
    if x in unique_list:
        return 'no_duplicates'
    else:
        return 'has_duplicates'
    
def identifyValidationRows(x, metadata_val):
    """ Identify Rows that represent validation data
    Input metadata_val: pandas data series of validation data
    """
    val_list = list(metadata_val)
    
    if str(x) in val_list:
        return 'val'
    else:
        return 'train'
    
def copyImagetoLabelFolder(metadata, metadata_train, metadata_val, figure_path, train_dir, val_dir):
    """Set up training directories with images
    Input metadata_train: pandas data series of training data
    Input metadata_val: pandas data series of validation data
    Input figure_path: general image directory 
    Input train_dir: training directory
    Input val_dir: validation directory
    """
    print('Copying images ...')
    train_list = list(metadata_train['image_id'])
    val_list = list(metadata_val['image_id'])
    metadata.set_index('image_id', inplace=True)
    for image_id in train_list:    
        fname = image_id + '.jpg'
        label = metadata.loc[image_id,'dx']
        src = '{}{}'.format(figure_path, fname)
        dst = '{}{}/{}'.format(train_dir, label, fname)
        shutil.copyfile(src, dst)
    for image_id in val_list:    
        fname = image_id + '.jpg'
        label = metadata.loc[image_id,'dx']
        src = '{}{}'.format(figure_path, fname)
        dst = '{}{}/{}'.format(val_dir, label, fname)
        shutil.copyfile(src, dst)
    print('... done copyint images !!!')

def dataAugmentation(class_list, data_path, train_dir, total_number_images=6000, target_size=(224,224), batch_size=50):
    '''
    Input class_list: list with class names
    Input data_path: general data directory
    Input train_dir: training directory
    Input total_number_images: total number of images that shall be created
    Input target_size: target size of image augmentatin
    Input batch_size: batch size of imaage augmentation
    '''
    print('Augmenting images ...')
    for img_class in class_list:
        print('Creating data for {} label'.format(img_class))
        aug_dir = '{}aug_dir/{}'.format(data_path, img_class)
        Path(aug_dir).mkdir(parents=True, exist_ok=True)
        img_dir = os.path.join(aug_dir, 'img_dir')
        Path(img_dir).mkdir(parents=True, exist_ok=True)
        img_list = os.listdir(train_dir + img_class)    
        for fname in img_list:
                src = os.path.join(train_dir + img_class, fname)
                dst = os.path.join(img_dir, fname)
                shutil.copyfile(src, dst)   
        datagen = ImageDataGenerator(
            rotation_range=180,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            #brightness_range=(0.9,1.1),
            fill_mode='nearest')
        aug_datagen = datagen.flow_from_directory(aug_dir,
                                               save_to_dir=train_dir + img_class,
                                               save_format='jpg',
                                                        target_size=target_size,
                                                        batch_size=batch_size)        
        num_files = len(os.listdir(img_dir))
        num_batches = int(np.ceil((total_number_images-num_files)/batch_size))
        for i in range(0,num_batches):
            imgs, labels = next(aug_datagen)    
        shutil.rmtree(aug_dir) 
    print('... done augmenting images !!!')

def checkDataVolume(path):
    """ Train Model
    Input model: defined model
    Input X_train: training data
    Input Y_train: training labels
    Input X_train: test data
    Input Y_train: test labels
    Return model: trained model
    Return history: training history
    """
    print(len(os.listdir('{}nv'.format(path))))
    print(len(os.listdir('{}mel'.format(path))))
    print(len(os.listdir('{}bkl'.format( path))))
    print(len(os.listdir('{}bcc'.format( path))))
    print(len(os.listdir('{}akiec'.format( path))))
    print(len(os.listdir('{}vasc'.format( path))))
    print(len(os.listdir('{}df'.format( path))))
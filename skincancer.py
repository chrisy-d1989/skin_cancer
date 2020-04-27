# -*- coding: utf-8 -*-
"""

modul: skin cancer detection
modul author: Christoph Doerr

"""

import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
import os.path
import shutil
#from selenium import webdriver #For Prediction
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, svm
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import utils as np_utils
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras import backend as K 
from keras.callbacks import ModelCheckpoint
from PIL import Image
import os
from glob import glob
import csv
import model_utils as model_utils
import utils_skincancer as utils
data_path = 'C:/backup/skin-cancer-mnist-ham10000/'
figure_path = '{}HAM10000_images_part_1/'.format(data_path)
train_dir = '{}pictures/train/'.format(data_path)
val_dir = '{}pictures/validation/'.format(data_path)
safe_model_path = 'C:/Users/cdoerr1/Desktop/CoronaAi/data/models/skincancer/'
checkpoint_path = 'C:/Users/cdoerr1/Desktop/CoronaAi/data/models/skincancer/checkpoints/'
lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}


imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                      for x in glob(os.path.join(data_path, '*', '*.jpg'))}

# if os.path.exists('{}features.npz'.format(data_path)):
#     print('Loading Data from features.npz ...')
#     features = np.load('{}features.npz'.format(data_path), allow_pickle=True)
#     labels = np.load('{}labels.npz'.format(data_path), allow_pickle=True)
#     features = features['arr_0']
#     labels = labels['arr_0']
#     print('... done loading Data from features.npz !!!')
# else:
print('Creating data ......')
metadata = pd.read_csv('{}HAM10000_metadata.csv'.format(data_path))
metadata['path'] = metadata['image_id'].map(imageid_path_dict.get)
metadata['cell_type'] = metadata['dx'].map(lesion_type_dict.get) 
metadata['cell_type_idx'] = pd.Categorical(metadata['cell_type']).codes
metadata['age'].fillna(metadata['age'].mean(), inplace=True)
# metadata['image'] =  metadata['path'].map(lambda x: np.asarray(image.load_img(x, target_size=(224,224,3))))
# features = np.asarray(metadata['image'].tolist())
labels = np.array(metadata['cell_type_idx'])
# np.savez_compressed('{}features.npz'.format(data_path), features)
# np.savez_compressed('{}labels.npz'.format(data_path), labels)
print('... done creating data !!!')

#check for duplicates in the data
metadata['duplicates'] = metadata['lesion_id']
metadata['duplicates'] = metadata['duplicates'].apply(utils.identify_duplicates, metadata=metadata['lesion_id'])

#create validation dataset without any duplicates
tmp = metadata[metadata['duplicates'] == 'no_duplicates']
y = tmp['dx']
_, metadata_val = train_test_split(tmp, test_size=0.2, random_state=101, stratify=y)

#create train dataset without any data from the validation data
metadata['train_or_val'] = metadata['image_id']
metadata['train_or_val'] = metadata['train_or_val'].apply(utils.identify_val_rows, metadata_val=metadata_val['image_id'])
metadata_train = metadata[metadata['train_or_val'] == 'train']

# print('Copying images ...')
# utils.copyImagetoLabelFolder(metadata, metadata_train, metadata_val, figure_path, train_dir, val_dir)
# print('... done copyint images !!!')
# print('Augmenting images ...')
# utils.dataAugmentation([*lesion_type_dict], data_path, train_dir)
# print('... done augmenting images !!!')

batch_size = 10
image_size=224
number_epochs = 10 
num_train_samples = len(metadata_train)
num_val_samples = len(metadata_val)
input_shape = (image_size, image_size, 3) 
train_steps = np.ceil(num_train_samples / batch_size)
val_steps = np.ceil(num_val_samples / batch_size)
datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input)

train_batches = datagen.flow_from_directory(train_dir,target_size=(image_size,image_size),batch_size=batch_size)
valid_batches = datagen.flow_from_directory(val_dir,target_size=(image_size,image_size),batch_size=batch_size)
test_batches = datagen.flow_from_directory(val_dir, target_size=(image_size,image_size), batch_size=1, shuffle=False)

model = keras.applications.resnet.ResNet50(include_top=True, weights=None, input_tensor=None, input_shape=input_shape, pooling=None, classes=7)
schedule = tf.keras.callbacks.LearningRateScheduler(model_utils.lr_schedule)

checkpoint = ModelCheckpoint('{}model.h5'.format(checkpoint_path), monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [schedule, checkpoint]
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print('Fitting model ...')
history = model.fit_generator(train_batches, steps_per_epoch=train_steps, validation_data=valid_batches,
                              validation_steps=val_steps, epochs=30, verbose=1, callbacks=callbacks_list)
# history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), epochs=number_epochs, callbacks=callbacks_list, validation_data = (x_test, y_test))
# new_model.fit(x_train, y_train, epochs=5, batch_size=50, callbacks=callbacks_list)
# model_utils.safeModel(model, safe_model_path, number_epochs, batch_size)
# model_utils.plotModelPerformance(history)

# checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
# callbacks_list = [checkpoint]
# new_model.fit(x_train, y_train, epochs=5, batch_size=50, callbacks=callbacks_list)

# image_list = os.listdir(figure_path)
# images = np.zeros((1,4))
# for i, image_path in enumerate(os.listdir(figure_path)):
#     image_array = image.load_img('{}{}'.format(figure_path, image_path), target_size=(224,224,3))
#     if i == 0:
#         images= np.expand_dims(np.array(image_array, dtype= float)/255, axis= 0)
#     else:
#         image_array= np.expand_dims(np.array(image_array, dtype= float)/255, axis= 0)
#         images= np.append(images, image_array, axis= 0)

# print('Preprocessing image data ...')
# features = tf.keras.applications.resnet50.preprocess_input(features)
# for i in range(len(features)):
#     features[i] = features[i].astype('float32')
#     features_normalize = np.true_divide(features[i], 255)
# print('... done preprocessing image data !!!')
# img_width, img_height = features.shape[1], features.shape[2]              

# x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

# y_train = np_utils.to_categorical(y_train, num_classes = 7)
# y_test = np_utils.to_categorical(y_test, num_classes = 7)
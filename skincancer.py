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
import tensorflow.keras
from tensorflow.keras import utils as np_utils
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D 
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras import backend as K 
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy
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
test_dir = '{}pictures/test/'.format(data_path)
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
metadata['train_or_val'] = metadata['train_or_val'].apply(utils.identifyValidationRows, metadata_val=metadata_val['image_id'])
metadata_train = metadata[metadata['train_or_val'] == 'train']

batch_size = 10
image_size= 224
number_epochs = 75 
num_train_samples = len(metadata_train)
num_val_samples = len(metadata_val)
num_test_samples = len(os.listdir(test_dir + 'all_clases'))
input_shape = (image_size, image_size, 3) 
train_steps = np.ceil(num_train_samples / batch_size)
val_steps = np.ceil(num_val_samples / batch_size)
test_steps = np.ceil(num_test_samples/1)
resnet = False
mobilenet = True
datagen = model_utils.preprocessingData(resnet=resnet, mobilenet=mobilenet)
# datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input)
# utils.copyImagetoLabelFolder(metadata, metadata_train, metadata_val, figure_path, train_dir, val_dir)
# utils.dataAugmentation([*lesion_type_dict], data_path, train_dir, total_number_images=10000, target_size=(224,224),\
#     batch_size=50)
 
train_batches = datagen.flow_from_directory(train_dir,target_size=(image_size,image_size),batch_size=batch_size)
valid_batches = datagen.flow_from_directory(val_dir,target_size=(image_size,image_size),batch_size=batch_size)
test_batches = datagen.flow_from_directory(val_dir, target_size=(image_size,image_size), batch_size=1, shuffle=False)

model = model_utils.defineModel(input_shape, num_classes=7, resnet50=resnet, mobilenet=mobilenet)

for layer in model.layers[:-25]:
    layer.trainable = False
for layer in model.layers[-25:]:
    layer.trainable = True
callbacks_list = model_utils.defineCallbacks(checkpoint_path, number_epochs, schedule=True, stopping=True, plateau=True,\
                                              checkpoint=False)
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[categorical_accuracy, model_utils.top_2_accuracy,\
                                                                            model_utils.top_3_accuracy])
# print('Fitting model ...')
history = model.fit_generator(train_batches, steps_per_epoch=train_steps, validation_data=valid_batches,
                                validation_steps=val_steps, epochs=number_epochs, initial_epoch=0, verbose=1, \
                                    callbacks=callbacks_list)
model_utils.safeModel(model, safe_model_path, number_epochs, batch_size, history)
# model = tf.keras.models.load_model('{}{}.h5'.format(safe_model_path, 'mobilnet_75_10'), \
#                                     custom_objects={'categorical_accuracy': categorical_accuracy,\
#                                                     'top_2_accuracy':  model_utils.top_2_accuracy,\
#                                                     'top_3_accuracy':  model_utils.top_3_accuracy})
# # test_generator.reset()
# val_loss, val_cat_acc, val_top_2_acc, val_top_3_acc = model.evaluate(test_batches, steps=num_val_samples)
# print('val_loss:', val_loss)
# print('val_cat_acc:', val_cat_acc)
# print('val_top_2_acc:', val_top_2_acc)
# print('val_top_3_acc:', val_top_3_acc)
# # # make a prediction
# predictions = model.predict(test_batches, steps=num_val_samples, verbose=1)
# print(predictions[2])
# print(predictions[309])
# print(predictions[1678])
# model_utils.plotHistogramm(predictions, valid_batches)
# history = pd.read_csv('{}{}_history.csv'.format(safe_model_path, 'resnet_75_10'))
# model_utils.plotModelPerformance(history)
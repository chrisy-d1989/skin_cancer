# -*- coding: utf-8 -*-
"""
modul: model utilities
modul author: Christoph Doerr

"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
import os
import tensorflow as tf
from keras import utils as np_utils
import matplotlib.pyplot as plt
import math
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator

def trainModel(model, X_train, Y_train, X_test, Y_test, batch_size=32, number_epochs=51, validation_split = 0.2, loss = 'sparse_categorical_crossentropy'):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    lrate = LearningRateScheduler(step_decay)
    callbacks_list = [lrate]
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    history = model.fit(X_train, Y_train, epochs=number_epochs, batch_size=batch_size, callbacks=callbacks_list, validation_data = (X_test, Y_test))
    # model.evaluate(X_test, Y_test)
    return (model, history)

def trainLSTMModel(model, X_train, Y_train, X_test, Y_test, batch_size, number_epochs=51, validation_split = 0.2, loss = 'sparse_categorical_crossentropy'):
    # model.compile(optimizer='adam', loss=loss, metrics=['accuracy'],)
    model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, Y_train, epochs=number_epochs, batch_size = batch_size)
    # model.evaluate(X_test, Y_test)
    return (model, history)

def splitData(stock, start_idx, end_idx, forcast_idx, ratio= 0.75):
    data_points = end_idx - start_idx
    data_points_train = int(data_points * ratio)
    train_data = stock.iloc[start_idx:(start_idx + data_points_train)].reset_index()
    test_data = stock.iloc[(start_idx + data_points_train + 1):end_idx].reset_index()
    predict_data = stock.iloc[end_idx:forcast_idx].reset_index()
    return (train_data, test_data, predict_data)

def getXY(train_data, test_data, predict_data):
    """Saving keras model
    Safes previous trained model
    Input safe_model_path: path to safe model 
    Input number_epochs: number of epochs model was trained with
    Input batch_size: batch size model was trained with
    Input model_name: defined model name(string), default value is none and a number is chosen
    """
    #toDo: use regex filter
    X_train = train_data.drop(['index', 'Date','daily_label', 'future_close'], axis = 1).to_numpy(dtype='float32')
    Y_train = train_data['daily_label'].to_numpy(dtype='float32')
    X_test = test_data.drop(['index', 'Date','daily_label', 'future_close'], axis = 1).to_numpy(dtype='float32')
    Y_test = test_data['daily_label'].to_numpy(dtype='float32')
    X_predict = predict_data.drop(['index', 'Date','daily_label', 'future_close'], axis = 1).to_numpy(dtype='float32')
    Y_predict = predict_data['daily_label'].to_numpy(dtype='float32')
    return (X_train, Y_train, X_test, Y_test, X_predict, Y_predict)

def prepareDataforLTSM(data, sample_length = 300, Y_data = False):
    number_samples = int(len(data)/sample_length)
    if number_samples == 0:
        print("Dataset to small for batch size")
    if Y_data:
        features = 1
    else:
        features = len(data[0]) 
    samples = np.zeros((number_samples, sample_length, features))  
    for j in range(0, number_samples):
        if Y_data:
            sample = data[j*sample_length : j*sample_length + sample_length]
            samples[j, :, 0] = sample
        else:    
            sample = data[j*sample_length : j*sample_length + sample_length, :]
            samples[j, :, :] = sample
    return (samples, number_samples)

def standardizeIndicators(stock):
    stock_std = stock.copy()
    for key, value in stock_std.iteritems():
        if key == 'Date' or 'index':
            continue
        elif (key == 'daily_label'):
                mean = 0
                std = 1
        else:
            mean = value.mean()
            std = value.std()
        stock_std.loc[:,key] = (stock.loc[:, key] - mean) / std
        if(stock_std[key].isnull().any().any()):
                print("Watch out, NANs in indicator data")
    return (stock_std)

def normalizeIndicators(stock):
    stock_std = stock.copy()
    for key, value in stock_std.iteritems():
        if key == 'Date' or key == 'index':
            continue
        else:
            x = stock_std[[key]].values.astype(float)
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(x)
            stock_std[key] = pd.DataFrame(x_scaled)
            stock_std[key] = stock_std[key].fillna(method='ffill')
            # if(stock_std[key].isnull().any().any()):
            #     print("Watch out, NANs in indicator data")
            # for index, row in stock_std.iterrows():
            #     if(int(stock_std[key].iloc[index])>1 or int(stock_std[key].iloc[index]<0)):
            #         print("Watch out, Normalization out of bounds")
    return (stock_std)

def safeModel(model, safe_model_path, number_epochs, batch_size, model_name=None):
    """Saving keras model
    Safes previous trained model
    Input safe_model_path: path to safe model 
    Input number_epochs: number of epochs model was trained with
    Input batch_size: batch size model was trained with
    Input model_name: defined model name(string), default value is none and a number is chosen
    """
    if model_name == None:
        model_name = len(os.listdir(safe_model_path)) + 1
    model.save('{}{}_{}_{}.h5'.format(safe_model_path, model_name, number_epochs, batch_size))
    print('safed model to {}{}_{}_{}.h5'.format(safe_model_path, model_name, number_epochs, batch_size))

def loadModel(safe_model_path, model_name):
    """Loading keras model
    Loads previous trained model
    Input safe_model_path: path to safed model 
    Input model name: model name of safed model
    Return: pretrained keras model
    """
    print('loading model to {}{}.h5'.format(safe_model_path, model_name))
    return tf.keras.models.load_model('{}{}.h5'.format(safe_model_path, model_name))

def defineCNN(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='elu', input_shape=(input_shape,)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1024, activation='elu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='elu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='elu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='elu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    return model

def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def dataAugmentation(data, resnet=False):
    if ~resnet:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range = 0.1, # Randomly zoom image 
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)  # randomly flip images
    else:
        datagen = ImageDataGenerator(preprocessing_function = tf.keras.applications.resnet.preprocess_input)
    data = datagen.fit(data)
    return data

def plotModelPerformance(model):
    """Plot Model Performace
    Plotting loss, validation loss, accuracy and validatoin accuracy over episodes
    Input: history of model fitting

    """
    fig = plt.figure(figsize=(10,8))
    ax0 = plt.subplot2grid((6, 1), (0, 0), rowspan=6)
    ax0.plot(model.history['loss'], label='loss', color='blue')
    ax0.plot(model.history['val_loss'], label='val loss', color='black')
    ax1 = ax0.twinx()
    ax1.plot(model.history['accuracy'], label= 'accuracy', color='orange')
    ax1.plot(model.history['val_accuracy'], label= 'accuracy', color='green')
    # plt.plot(model_validation.history['loss'], label='val_loss')
    ax0.set_title('model loss and accurcy')
    ax0.set_ylabel('loss')
    ax0.set_xlabel('epoch')
    ax0.set_ylim([0,2])
    ax1.set_ylim([0,2])
    lines, labels = ax0.get_legend_handles_labels()
    lines2, labels2 = ax1.get_legend_handles_labels()
    ax0.legend(lines + lines2, labels + labels2, loc=1)
    plt.show()

# model = tf.keras.Sequential([
#     tf.keras.layers.LSTM(500, input_shape=(X_train.shape[1], X_train.shape[2]), batch_size = batch_size, dropout=0.2, recurrent_dropout=0.1, return_sequences=True),
#     tf.keras.layers.Dense(2, activation='softmax')
# ])

# model = tf.keras.Sequential([
#     tf.keras.layers.LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), stateful=True, batch_size = batch_size, dropout=0.2, recurrent_dropout=0.1, return_sequences=True),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.LSTM(1028, dropout=0.2, stateful=True, recurrent_dropout=0.1, return_sequences=True),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.LSTM(512, dropout=0.2, stateful=True, recurrent_dropout=0.1, return_sequences=True),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.LSTM(256, dropout=0.2, stateful=True, recurrent_dropout=0.1, return_sequences=True),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.LSTM(128, dropout=0.2, stateful=True, recurrent_dropout=0.1, return_sequences=True),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.LSTM(64, dropout=0.2, stateful=True, recurrent_dropout=0.1, return_sequences=True),
#     tf.keras.layers.Dense(2,activation='softmax')
# ])

# predict_model = tf.keras.Sequential([
#     tf.keras.layers.LSTM(64, input_shape=(X_predict.shape[1], X_predict.shape[2]), batch_size = batch_size_predict, dropout=0.2, recurrent_dropout=0.1, return_sequences=True),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.LSTM(1028, dropout=0.2, stateful=True, recurrent_dropout=0.1, return_sequences=True),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.LSTM(512, dropout=0.2, stateful=True, recurrent_dropout=0.1, return_sequences=True),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.LSTM(256, dropout=0.2, stateful=True, recurrent_dropout=0.1, return_sequences=True),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.LSTM(128, dropout=0.2, stateful=True, recurrent_dropout=0.1, return_sequences=True),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.LSTM(64, dropout=0.2, stateful=True, recurrent_dropout=0.1, return_sequences=True),
#     tf.keras.layers.Dense(2,activation='softmax')
# ])



# # copy weights
# old_weights = model.get_weights()
# predict_model.set_weights(old_weights)
# model_prediction = predict_model.predict(X_predict)
# apa['prediction'] = np.full((len(apa['Adj Close']),1), -1)
# apa.loc[(end_idx+1):forcast_idx, 'prediction'] = model_prediction[0,:,0]

# print(model_prediction[0,:,1])
# print(apa['daily_label'].iloc[(end_idx+1):forcast_idx])
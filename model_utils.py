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
from tensorflow.keras import utils as np_utils
import matplotlib.pyplot as plt
import math
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy


def trainModel(model, X_train, Y_train, X_test, Y_test, batch_size=32, number_epochs=51, validation_split = 0.2,\
               loss = 'sparse_categorical_crossentropy'):
    """ Train Model
    Input model: defined model
    Input X_train: training data
    Input Y_train: training labels
    Input X_train: test data
    Input Y_train: test labels
    Return model: trained model
    Return history: training history
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    lrate = LearningRateScheduler(lr_schedule)
    callbacks_list = [lrate]
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    history = model.fit(X_train, Y_train, epochs=number_epochs, batch_size=batch_size, callbacks=callbacks_list, validation_data = (X_test, Y_test))
    # model.evaluate(X_test, Y_test)
    return (model, history)

def trainLSTMModel(model, X_train, Y_train, X_test, Y_test, batch_size, number_epochs=51, validation_split = 0.2,\
                   loss = 'sparse_categorical_crossentropy'):
    """ Train LSTM Model
    Input model: defined model
    Input X_train: training data
    Input Y_train: training labels
    Input X_train: test data
    Input Y_train: test labels
    Return model: trained model
    Return history: training history
    """
    # model.compile(optimizer='adam', loss=loss, metrics=['accuracy'],)
    model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, Y_train, epochs=number_epochs, batch_size = batch_size)
    # model.evaluate(X_test, Y_test)
    return (model, history)

def splitData(stock, start_idx, end_idx, forcast_idx, ratio= 0.75):
    """ Split data set into train, test and predict data sets
    Input stock: pandas data series
    Input start_idx: index of the start date
    Input end_idx: index of the end date
    Input forcast_idx: index of the forcasted date
    Input ratio: ratio of train and test data, default value is 0.75
    return stock_std: train, test and predict data set
    """
    data_points = end_idx - start_idx
    data_points_train = int(data_points * ratio)
    train_data = stock.iloc[start_idx:(start_idx + data_points_train)].reset_index()
    test_data = stock.iloc[(start_idx + data_points_train + 1):end_idx].reset_index()
    predict_data = stock.iloc[end_idx:forcast_idx].reset_index()
    return (train_data, test_data, predict_data)

def getXY(train_data, test_data, predict_data):
    """ Split data set into labels and train data
    Input stock: pandas data series
    return stock_std: train, test and predict data set, train, test, predict labels
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
    """ Prepare Data for LSTM Model training
    Input data:
    Return samples:
    Return number_samples:
    """
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
    """ Standardize dataset of indicators
    Input stock: pandas data series
    return stock_std: standardized pandas series
    """
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
    """ Normalize dataset of indicators
    Input stock: pandas data series
    return stock_std: normalized pandas series
    """
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

def safeModel(model, safe_model_path, number_epochs, batch_size, history, model_name=None):
    """Saving keras model
    Safes previous trained model and training history
    Input safe_model_path: path to safe model 
    Input number_epochs: number of epochs model was trained with
    Input batch_size: batch size model was trained with
    Input history: training history
    Input model_name: defined model name(string), default value is none and a number is chosen
    """
    if model_name == None:
        model_name = len(os.listdir(safe_model_path)) + 1
    model.save('{}{}_{}_{}.h5'.format(safe_model_path, model_name, number_epochs, batch_size))
    print('safed model to {}{}_{}_{}.h5'.format(safe_model_path, model_name, number_epochs, batch_size))
    hist_df = pd.DataFrame(history.history)
    hist_csv_file = '{}{}_{}_{}_history.csv'.format(safe_model_path, model_name, number_epochs, batch_size)
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)
        
def loadModel(safe_model_path, model_name):
    """Loading pretrained keras model
    Input safe_model_path: path to safed model 
    Input model_name: model name of safed model
    Return: pretrained keras model
    """
    print('loading model from {}{}.h5'.format(safe_model_path, model_name))
    return tf.keras.models.load_model('{}{}.h5'.format(safe_model_path, model_name))

def defineModel(input_shape, X_train=None, num_classes=7, batch_size=None, stockModel1=False, stockModelLSTM1=False, \
                stockModelLSTM2=False, resnet50=False, mobilenet=False, randomnet=False):
    """Define keras model architecture
    Input input_shape: input shape of training data, shape depends on the model 
    Input num_classes: number of classes
    Input batch_size: batch size of training
    Input stockModel1: Boolean, True if you want to use stockModel1
    Input stockModelLSTM1: Boolean, True if you want to use stockModelLSTM1
    Input stockModelLSTM2: Boolean, True if you want to use stockModelLSTM2
    Input resnet50: Boolean, True if you want to use resnet50
    Input mobilenet: Boolean, True if you want to use mobilenet
    Input randomnet: Boolean, True if you want to use randomnet
    Return: model
    """
    if stockModel1:
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
        
    if stockModelLSTM1: 
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(500, input_shape=(X_train.shape[1], X_train.shape[2]), batch_size = batch_size,\
                                 dropout=0.2, recurrent_dropout=0.1, return_sequences=True),
            tf.keras.layers.Dense(2, activation='sigmoid')
        ])
        
    if stockModelLSTM2:
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), stateful=True, batch_size = batch_size,\
                                 dropout=0.2, recurrent_dropout=0.1, return_sequences=True),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.LSTM(1028, dropout=0.2, stateful=True, recurrent_dropout=0.1, return_sequences=True),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.LSTM(512, dropout=0.2, stateful=True, recurrent_dropout=0.1, return_sequences=True),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.LSTM(256, dropout=0.2, stateful=True, recurrent_dropout=0.1, return_sequences=True),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.LSTM(128, dropout=0.2, stateful=True, recurrent_dropout=0.1, return_sequences=True),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.LSTM(64, dropout=0.2, stateful=True, recurrent_dropout=0.1, return_sequences=True),
            tf.keras.layers.Dense(2,activation='sigmoid')
        ])

    if resnet50:
        model = tf.keras.applications.resnet.ResNet50(include_top=True, weights=None, input_tensor=None,\
                                                      input_shape=input_shape, pooling=None, classes=num_classes)
        x = model.layers[-6].output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        model = tf.keras.models.Model(inputs=model.input, outputs=predictions)

    if mobilenet:
        model = tf.keras.applications.mobilenet.MobileNet(input_shape=input_shape, alpha=1.0, depth_multiplier=1, dropout=0.2,\
                                                       include_top=True, weights=None, input_tensor=None, pooling=None,\
                                                           classes=num_classes)
        x = model.layers[-6].output
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        model = tf.keras.models.Model(inputs=model.input, outputs=predictions)
    
    if randomnet:
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3),activation='relu',padding = 'Same',input_shape=input_shape),
            tf.keras.layers.Conv2D(32,kernel_size=(3, 3), activation='relu',padding = 'Same',),
            tf.keras.layers.MaxPool2D(pool_size = (2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu',padding = 'Same'),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu',padding = 'Same'),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.40),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

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
    return model

def top_3_accuracy(y_true, y_pred):
    """Calculate Top3 Accuracy
    return: top3 accuracy
    """
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def top_2_accuracy(y_true, y_pred):
    """Calculate Top2 Accuracy
    return: Top2 accuracy
    """
    return top_k_categorical_accuracy(y_true, y_pred, k=2)



def preprocessingData(resnet=False, mobilenet=False):
    """Preprocessing Data for model training
    Input resnet: boolean, if you want to preprocess for resnet model
    Input mobilenet: boolean, if you want to preprocess for mobilenet model
    Return datagen: data generator
    """
    if resnet:
        datagen = ImageDataGenerator(preprocessing_function = tf.keras.applications.resnet.preprocess_input)
    if mobilenet:
        datagen = ImageDataGenerator(preprocessing_function = tf.keras.applications.mobilenet.preprocess_input)
    return datagen

def checkpointModel(checkpoint_path, model_name=None):
    """Saving keras model checkpoints while fitting
    Input checkpoint_path: path to checkpoints 
    Return : safing checkpoint
    """
    if model_name == None:
        model_name = len(os.listdir(checkpoint_path)) + 1
    return ModelCheckpoint('{}{}.h5'.format(checkpoint_path, model_name), monitor='categorical_accuracy', verbose=1,\
                           save_best_only=True, mode='min', save_weights_only=False)

def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    input epoch: number of epochs
    return lr: learning rate
    """
    lr = 1e-4
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

def defineCallbacks(checkpoint_path, epoch, schedule=False, stopping=False, plateau=False, checkpoint=False):
    """Plot Model Performace
    input checkpoint_path: path where checkpoints shall be safed
    input schedule: boolean if a learning rate scheduler shall be used while fitting
    input stopping: boolean if learning shall be stopped if there is no improvement detected
    input plateau: boolean if learning rate shall be reduced if learning hits a plateau
    input checkpoint: boolean if checkpoints shall be safed
    return callbacks: list of callbacks
    """   
    schedule = LearningRateScheduler(lr_schedule)
    stopping = EarlyStopping(monitor='categorical_accuracy', min_delta=1e-4, patience=10, verbose=0, mode='auto', baseline=None, \
                             restore_best_weights=True)
    learning_rate_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5,verbose=0, mode='auto', \
                                              min_delta=0.0001, min_lr=1e-6)
    checkpoint = checkpointModel(checkpoint_path)
    callbacks = []
    if schedule:
        callbacks.append(schedule)
    if plateau:
        callbacks.append(stopping)
    if schedule:
        callbacks.append(learning_rate_plateau)
    if checkpoint:
        callbacks.append(checkpoint)   
    return callbacks

def plotModelPerformance(model):
    """Plot Model Performace
    Plotting loss, validation loss, accuracy and validatoin accuracy over episodes
    Input: history of model fitting
    """
    fig = plt.figure(figsize=(10,8))
    ax0 = plt.subplot2grid((6, 1), (0, 0), rowspan=6)
    ax0.plot(model['loss'], label='loss', color='blue')
    ax0.plot(model['val_loss'], label='val loss', color='black')
    ax1 = ax0.twinx()
    ax1.plot(model['categorical_accuracy'], label= 'categorical_accuracy', color='orange')
    ax1.plot(model['val_categorical_accuracy'], label= 'val_categorical_accuracy accuracy', color='green')
    ax0.set_title('model loss and accurcy')
    ax0.set_ylabel('loss')
    ax0.set_xlabel('epoch')
    # ax0.set_ylim([0,2])
    # ax1.set_ylim([0,2])
    lines, labels = ax0.get_legend_handles_labels()
    lines2, labels2 = ax1.get_legend_handles_labels()
    ax0.legend(lines + lines2, labels + labels2, loc=1)
    plt.show()

def plotHistogramm(data, valid_batches):
    """Plot Prediction Histogramm
    Plotting probabilites of predicted classes
    Input: np array of predicted data
    Input valid_batches: dictonary of validation batches wit class names
    """
    labels = [*valid_batches.class_indices]
    x = np.arange(len(labels))  # the label locations
    width = 0.3  # the width of the bars
    fig, ax = plt.subplots(1,1, figsize=(12,3))
    # for i in range(3):
    #     ax.bar(x + i*width, data[i], width, label=i)
    ax.bar(x, data[2], width, label=0, color='green')
    ax.bar(x + width, data[309], width, label=1, color='blue')
    ax.bar(x + 2*width, data[1678], width, label=2, color='red')
    ax.set_ylabel('Scores')
    ax.set_title('Scores by picture')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

# # copy weights
# old_weights = model.get_weights()
# predict_model.set_weights(old_weights)
# model_prediction = predict_model.predict(X_predict)
# apa['prediction'] = np.full((len(apa['Adj Close']),1), -1)
# apa.loc[(end_idx+1):forcast_idx, 'prediction'] = model_prediction[0,:,0]

# print(model_prediction[0,:,1])
# print(apa['daily_label'].iloc[(end_idx+1):forcast_idx])
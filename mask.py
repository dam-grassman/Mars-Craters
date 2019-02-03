# -*- coding: utf-8 -*-
"""
U-Net Neural Network in order to build a mask out of the crater images
"""

from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, concatenate, Input, Dropout, UpSampling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint#, LearningRateScheduler
#from keras import backend as keras

from math import ceil
from utils import divide_image
from generators import get_train_valid_generators


def unet(pretrained_weights = None,input_size = (112,112,1)):
    
    ''' U-Net Neural Network inspired by U-Net: Convolutional Networks for Biomedical Image Segmentation.
    implemented in https://github.com/zhixuhao/unet/blob/master/model.py'''
    
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model


def get_mask_model(X_train, y_train):
    
    ''' Train the mask U-net model. We call an image generator and then pass it 
    into the model '''
    
    BATCH_SIZE = 2
    
    l_train = []
    for i in range(X_train.shape[0]):
        l_train.extend(divide_image(X_train[i], y_train[i]))

    train_generator, val_generator, n_train_samples, n_val_samples = \
            get_train_valid_generators(l_train, batch_size=BATCH_SIZE)

    # create the callbacks to get during fitting
    callbacks = []
    callbacks.append( ModelCheckpoint('./UNET_weights_best.h5',
                        monitor='val_loss', verbose=0,
                        save_best_only=True, save_weights_only=True,
                        mode='auto', period=1))
    # add early stopping
    #callbacks.append(EarlyStopping(monitor='val_loss', min_delta=0.001,
    #                               patience=10, verbose=1))

    # reduce learning-rate when reaching plateau
    #callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                       #patience=5, epsilon=0.001,
                                       #cooldown=2, verbose=1))
    model = unet()

    model.fit_generator(
       generator=train_generator,
        steps_per_epoch=ceil(n_train_samples / BATCH_SIZE),
        epochs=3,
        callbacks=callbacks,
        validation_data=val_generator,
        validation_steps=ceil(n_val_samples / BATCH_SIZE))
    
    return model


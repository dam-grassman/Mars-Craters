# -*- coding: utf-8 -*-

import random
import numpy as np
from utils import fix_labels, get_image_augmented
from keras.preprocessing.image import ImageDataGenerator

def CraterGenerator(X,y, indices=None):
    
    ''' Generate differents images with craters '''
    
    if indices is None : 
        indices = [i for i in range(X.shape[0])]

    datagen = ImageDataGenerator(featurewise_center=False,  featurewise_std_normalization=False,  rotation_range=20,
        width_shift_range=0, height_shift_range=0, horizontal_flip=True, vertical_flip=True,  zoom_range = [0.4,1.],
        rescale=1)
    
    datagen.fit(X.reshape(X.shape[0],224,224,1))
    batch_id = 0
    batch_size = 50
    
    while True:
        # First we add the craters as layers of our image
        X_augmented = np.array([get_image_augmented(X[ind], y[ind][:4]) for ind in 
                                indices[batch_id*batch_size: (batch_id+1)*batch_size]])
        # Alongside their labels
        y_train_ind = [y[ind][:4] for ind in indices[batch_id*batch_size:(batch_id+1)*batch_size]]
        
        
        if len(y_train_ind) != batch_size:
            new_index = np.random.choice(indices, batch_size - len(y_train_ind))
            X_augmented = np.concatenate([X_augmented, np.array([get_image_augmented(X[ind], y[ind][:4]) for ind in 
                               new_index])], axis=0)
            y_train_ind.extend([y[ind][:4] for ind in new_index])

        # We use the data_generator from keras to generate a batch of images
        for x_b, y_b in datagen.flow(X_augmented, y_train_ind, batch_size=batch_size):

            # The first layer correspond to the real image
            X_batch = [np.concatenate([x_b[i][:,:,0].reshape(224,224,1),(x_b[i][:,:,1] + x_b[i][:,:,2] + x_b[i][:,:,3]
                                      +x_b[i][:,:,4]).reshape((224,224,1))], axis = 2) for i in range(min(batch_size,len(x_b)))] #[::,::,0]
            # We get the labels from the other layers
            y_batch = [fix_labels(x_b[i][::,::,:len(y_b[i])+1]) for i in range(min(batch_size,len(x_b)))]
            # And we take squares instead of circles 
           
            break
            

        batch_id = (batch_id+1)%((max(len(indices)//batch_size,1)))
        if batch_id == 0 : 
            random.shuffle(indices)

        yield X_batch, y_batch
        
def get_train_valid_generators(l, batch_size=2, valid_ratio=0.1):
    
    ''' Get the train and validation batches generator '''

    nb_valid = int(valid_ratio * len(l))
    nb_train =  len(l) - nb_valid
    indices = np.arange(len(l))
    train_indices = indices[0:nb_train]
    valid_indices = indices[nb_train:]
    
    l_train = [l[i] for i in train_indices]
    
    l_train = sorted(l_train, key = lambda tup : np.sum(tup[1]), reverse=True )
    #start = [l_train[int(len(l_train)/2)]]
    #print('Start', len(start))
    #l_train = l_train[:int(len(l_train)/2)] + l_train[int(len(l_train)/2)+1:]
    
    #start = l_train[int(3*len(l_train)/4):int(3*len(l_train)/4)+100]
    #print('Start', len(start))
    #l_train = l_train[:int(3*len(l_train)/4)] + l_train[int(3*len(l_train)/4)+100:]
    
    start = l_train[-20:]
    print('Start', len(start))
    l_train = l_train[:-20]
    print('Len',len(l_train))
    
    np.random.shuffle(l_train)
    l_train = start + l_train
    
    print(np.sum(l_train[0][1]), l_train[0][1] )
    print('SORTED')
    
    def _get_generator(l, batch_size=batch_size):
        while 1 :
            for i in range(0,len(l)-batch_size,batch_size):
                yield (np.array([l[i+a][0] for a in range(batch_size)]).reshape((batch_size,112,112,1)),
                       np.array([l[i+a][1] for a in range(batch_size)]).reshape((batch_size,112,112,1)))
    
    gen_train = _get_generator(l_train, batch_size=batch_size)
    gen_valid = _get_generator([l[i] for i in valid_indices], batch_size=batch_size)
    return gen_train, gen_valid, nb_train, nb_valid
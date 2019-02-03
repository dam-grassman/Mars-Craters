# -*- coding: utf-8 -*-
"""
Add some comments here
"""

import numpy as np 
from sklearn.model_selection import train_test_split
from keras.applications import vgg16
from keras.models import Model
from sklearn.ensemble import RandomForestClassifier

from crater_generator import CraterGenerator
from utils import reshape_image, get_dictionnary_craters,\
                    verify, merge_dic, get_random_box
                    #get_crater,dist,construct_mask,divide_image, construct_circle_plan\
                    #get_image_augmented, get_circle_from_array, fix_labels


def get_non_craters(array_x, array_y, default_radius=None):
    
    ''' Get non-craters images from images that don't contain (supposely)
    any craters '''
    
    dic_size = {}
    for ind in range(array_x.shape[0]):
        if len(array_y[ind])!=0 : 
            continue
        if default_radius == None : 
            list_craters = [get_random_box(array_x[ind], size) for size in [28,38,48,58,68]*20]
        else : 
            list_craters = [get_random_box(array_x[ind], size) for size in [default_radius, default_radius, default_radius, 
                                                                            default_radius, default_radius]]

        for image in list_craters:
            dic = dic_size.get(image.shape[0], [])
            dic.append(image)
            dic_size[image.shape[0]] = dic
    return dic_size

def get_model_by_size(size, dic_size_crater,dic_size_non_crater,  non_crater_factor=4):
    
    ''' Get RF classifier according to a given size '''
    
    siz_e = min(58,max(48,size))
    
    X_train_crater = [reshape_image(dic_size_crater[size][i], siz_e, 3) for i in range(len(dic_size_crater[size]))]
    y_train_crater = [1 for i in range(len(dic_size_crater[size]))]
    X_train_crater.extend([reshape_image(dic_size_non_crater[size][i], siz_e, 3) for i in range(len(dic_size_crater[size])*non_crater_factor)])
    y_train_crater.extend([0 for i in range(len(dic_size_crater[size])*non_crater_factor)])
    
    vgg_model = vgg16.VGG16(weights='imagenet', input_shape=(siz_e,siz_e,3), include_top=False)
    intermediate_layer_model = Model(inputs=vgg_model.input,
                                 outputs=vgg_model.layers[-1].output)
    
    step = 2000
    intermediate_output = intermediate_layer_model.predict(np.array(X_train_crater[:step]))
    for i in range(step,len(X_train_crater), step):
        intermediate_output = np.concatenate([intermediate_output,intermediate_layer_model.predict(np.array(X_train_crater[i:i+step]))])
    intermediate_output = intermediate_output.reshape((intermediate_output.shape[0],512))   
    #print(intermediate_output.shape)
    X_train_, X_test_, y_train_, y_test_ = train_test_split(intermediate_output, y_train_crater, test_size=0.3)
    
    rdf = RandomForestClassifier(n_estimators=400, class_weight = {0:1,1:2},max_depth = 15, min_samples_leaf=10, n_jobs = -1, min_samples_split=5)
    rdf.fit(X_train_, y_train_)
    
    #print(confusion_matrix(y_test_, rdf.predict(X_test_)))
    return rdf


def correct_dimension(img, label):
    
    ''' Correct the dimension of a crater-image. Sometimes when the crater is too close
    from the edge of the original image, the size is a bit truncated and we use a small
    trick to correct it with an allowed dimension. Basically we use the mirror of the edge
    on the smallest side. '''
    
    if img.shape[0] == img.shape[1]:
        return img
    else : 
        _,x,y,_ = label 
        print('Correstion')
        max_dim = max(img.shape[0], img.shape[1])
        new_img = np.zeros((max_dim,max_dim))
        #img = img[:,:,0]
        #print(img.shape, (img.shape[1]-img.shape[0]))
        
        if img.shape[1]<img.shape[0] and y > 112 :
            #print((img.shape[0]-img.shape[1]))
            new_img[:,:img.shape[1]] = img
            new_img[:,img.shape[1]:] = np.flip(img, axis=1)[:,:(img.shape[0]-img.shape[1])]
            
        if img.shape[1]<img.shape[0] and y < 112 :
            
            new_img[:,(img.shape[0]-img.shape[1]):] = img
            new_img[:,:(img.shape[0]-img.shape[1])] = np.flip(img, axis=1)[:,-(img.shape[0]-img.shape[1]):]
        
        elif img.shape[0]<img.shape[1] and x < 112 :
            
            new_img[(img.shape[1]-img.shape[0]):,:] = img
            new_img[:(img.shape[1]-img.shape[0]),:] = np.flip(img, axis=0)[-(img.shape[1]-img.shape[0]):,:]
            
        elif img.shape[0]<img.shape[1] and x > 112 :
    
            new_img[:img.shape[0],:] = img
            new_img[img.shape[0]:,:] = np.flip(img, axis=0)[:(img.shape[1]-img.shape[0]),:]
        else : 
            pass
        
        return new_img

def get_boxes_from_circle(img, circle):
    
    ''' From a given image, we select the parts containing craters, sorted by
    allowed sizes. This is a bit long, i need to make it more concise.'''
    
    boxes_28 = []
    boxes_38 = []
    boxes_48 = []
    boxes_58 = []
    boxes_68 = []

    indices_28 = []
    indices_38 = []
    indices_48 = []
    indices_58 = [] 
    indices_68 = []
    
    for acc, y,x,r in circle : 
        if (r - r%5 + 9)*2 < 29 :
            
            crater_img = reshape_image(correct_dimension(img[max(0,x-(r - r%5 + 9)):min(x+(r - r%5 + 9),224),
                                  max(0,y-(r - r%5 + 9)):min(y+(r - r%5 + 9),224)], (acc, x,y,r)),48,3)
            
            if crater_img.shape == (48,48,3):
                boxes_28.append(crater_img)
                indices_28.append((acc, x,y,r))
                
        elif   (r - r%5 + 9)*2 == 38:
            
            crater_img = reshape_image(correct_dimension(img[max(0,x-(r - r%5 + 9)):min(x+(r - r%5 + 9),224),
                                  max(0,y-(r - r%5 + 9)):min(y+(r - r%5 + 9),224)], (acc, x,y,r)),48,3)
            
            if crater_img.shape == (48,48,3):
                boxes_38.append(crater_img)
                indices_38.append((acc, x,y,r))
                
        elif   (r - r%5 + 9)*2 == 48:
            
            crater_img = reshape_image(correct_dimension(img[max(0,x-(r - r%5 + 9)):min(x+(r - r%5 + 9),224),
                                  max(0,y-(r - r%5 + 9)):min(y+(r - r%5 + 9),224)], (acc, x,y,r)),58,3)
            
            if crater_img.shape == (58,58,3):
                boxes_48.append(crater_img)
                indices_48.append((acc, x,y,r))
                
        elif   (r - r%5 + 9)*2 == 58:
            
            crater_img = reshape_image(correct_dimension(img[max(0,x-(r - r%5 + 9)):min(x+(r - r%5 + 9),224),
                                  max(0,y-(r - r%5 + 9)):min(y+(r - r%5 + 9),224)], (acc, x,y,r)),58,3)
            
            if crater_img.shape == (58,58,3):   
                boxes_58.append(crater_img)
                indices_58.append((acc, x,y,r))
            else : 
                print('Weird')
                
        elif   (r - r%5 + 9)*2 == 68:
            
            crater_img = reshape_image(correct_dimension(img[max(0,x-(r - r%5 + 9)):min(x+(r - r%5 + 9),224),
                                  max(0,y-(r - r%5 + 9)):min(y+(r - r%5 + 9),224)], (acc, x,y,r)),58,3)
            if crater_img.shape == (58,58,3):   
                boxes_68.append(crater_img)
                indices_68.append((acc, x,y,r))
            else : 
                print('Weird')
        else :
            pass


    return (boxes_28,indices_28),  (boxes_38,indices_38), \
            (boxes_48,indices_48), (boxes_58,indices_58), (boxes_68,indices_68)
            
def predict_circle(b28, b38, b48, b58, b68, rdf_28, rdf_38, rdf_58,intermediate_layer_model_48, intermediate_layer_model_58 ):
    print('28')
    for j,im in enumerate(b28[0][::-1]):
        try : 
            intermediate_output = intermediate_layer_model_48.predict(im.reshape((1,48,48,3)))
            pred =rdf_28.predict_proba(intermediate_output.reshape(1,512))
            if  np.argmax(pred)==0 : 
                b28[1].pop(len(b28[0])-j-1)
        except : 
            #print(im.shape)
            b28[1].pop(len(b28[0])-j-1)
        #print('NEXT')
    print('38')
    for j,im in enumerate(b38[0][::-1]):
        try : 
            intermediate_output = intermediate_layer_model_48.predict(im.reshape((1,48,48,3)))
            pred = rdf_38.predict_proba(intermediate_output.reshape(1,512))
            if  np.argmax(pred)==0 : 
                b38[1].pop(len(b38[0])-j-1)
        except : 
            #print(im.shape)
            b38[1].pop(len(b38[0])-j-1)
    print('48')

#     for j,im in enumerate(b48[0][::-1]):
#         try : 
#             intermediate_output = intermediate_layer_model_48.predict(im.reshape((1,48,48,3)))
#             pred = rdf_48.predict_proba(intermediate_output.reshape(1,512))
#             if  np.argmax(pred)==0 : 
#                 b48[1].pop(len(b48[0])-j-1)
#         except : 
#             #print(im.shape)
#             b48[1].pop(len(b48[0])-j-1)
#     print('58')
    for j,im in enumerate(b48[0][::-1]):
        try : 
            intermediate_output = intermediate_layer_model_58.predict(im.reshape((1,58,58,3)))
            pred = rdf_58.predict_proba(intermediate_output.reshape(1,512))
            if  np.argmax(pred)==0 : 
                b48[1].pop(len(b48[0])-j-1)
        except : 
            #print(im.shape)
            print('Shape 48')
            b48[1].pop(len(b48[0])-j-1)
    print('58')

    for j,im in enumerate(b58[0][::-1]):
        try : 
            intermediate_output =intermediate_layer_model_58.predict(im.reshape((1,58,58,3)))
            pred =rdf_58.predict_proba(intermediate_output.reshape(1,512))
            if  np.argmax(pred)==0 : 
                b58[1].pop(len(b58[0])-j-1)
        except : 
            #print(im.shape)
            print('Shape 58')
            b58[1].pop(len(b58[0])-j-1)
            
    print('68')
    for j,im in enumerate(b68[0][::-1]):
        try : 
            intermediate_output =intermediate_layer_model_58.predict(im.reshape((1,58,58,3)))
            pred = rdf_58.predict_proba(intermediate_output.reshape(1,512))
            if  np.argmax(pred)==0 : 
                b68[1].pop(len(b68[0])-j-1)
        except : 
            #print(im.shape)
            print('Shape 68')
            b68[1].pop(len(b68[0])-j-1)
#     for j,im in enumerate(b68[0][::-1]):
#         try : 
#             intermediate_output =intermediate_layer_model_58.predict(im.reshape((1,58,58,3)))
#             pred = rdf_68.predict_proba(intermediate_output.reshape(1,512))
#             if  np.argmax(pred)==0 : 
#                 b68[1].pop(len(b68[0])-j-1)
#         except : 
#             #print(im.shape)
#             b68[1].pop(len(b68[0])-j-1)

    return  b28[1] + b38[1] + b48[1] + b58[1] + b68[1]


def fit_classif(X_train, y_train):
    
    ''' Train the 3 differents RF classifiers. We use a CraterGenerator
    in order to get more images of craters. '''

    dic = get_dictionnary_craters(X_train, y_train)
    for key in dic :
        if (key-8)%10 == 0 :
            print('Shape ', key,':', len(dic[key]), 'craters')

    j = 0
    #for X, Y in CraterGenerator(X_train, y_train):
    for X_, Y_ in CraterGenerator(X_train[[i for i in range(X_train.shape[0]) if verify(y_train[i]) == True ]],\
                                y_train[[i for i in range(X_train.shape[0]) if verify(y_train[i]) == True]]):
        dic2 = get_dictionnary_craters(np.array(X_)[:,:,:,0].reshape((len(X_),224,224)), Y_)
        dic = merge_dic(dic,dic2)
        j+=1
        #print(i)
        if j == 100 : 
            break

    for key in dic :
        if (key-8)%10 == 0 :
            print('Shape ', key,':', len(dic[key]), 'craters')

    dic_size_non_crater = get_non_craters(X_train, y_train,None)
    for key in dic_size_non_crater :
        if (key-8)%10 == 0 :
            print('Shape ', key,':', len(dic_size_non_crater[key]), 'craters')

    l = dic[48] +  dic[58] +  dic[68]
    dic[58] = l 

    rdf_58 = get_model_by_size(58,dic,dic_size_non_crater,3)
    rdf_38 = get_model_by_size(38,dic,dic_size_non_crater,3)
    rdf_28 = get_model_by_size(28,dic,dic_size_non_crater,3)

    del dic, dic_size_non_crater
    
    return rdf_28, rdf_38, rdf_58
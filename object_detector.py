# -*- coding: utf-8 -*-
"""
Object Detector Class with a fit and predict instances used in the RAMP plateform
Needs a bit of cleaning though
"""

import numpy as np 
#from keras import backend as keras

from keras.applications import vgg16
from keras.models import Model

from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny

from mask import get_mask_model #,unet
from metrics import NMS_size #, NMS, iou
from classification import fit_classif, get_boxes_from_circle
#from crater_generator import CraterGenerator, get_train_valid_generators
#from utils import reshape_image, get_crater, get_dictionnary_craters, dist, \
#                    construct_mask, divide_image, construct_circle_plan,\
#                    get_image_augmented, get_circle_from_array, fix_labels, \
#                    verify, merge_dic, get_random_box


class ObjectDetector(object):
    
    def __init__(self):
        self.rdf_28 = None
        self.rdf_38 = None
        self.rdf_58 = None
        self.mask = None
                 
    def fit(self, X, y):
        
        ''' We need to fit the mask u-net model and the random forests classifiers
        for the different sizes of craters '''
        
        self.mask = get_mask_model(X, y)
        self.rdf_28, self.rdf_38, self.rdf_58 = fit_classif(X, y)
    
    def predict(self, X, batch_size = 2000):
        
        ''' First of all we construct the mask of each image. Rmq : as I have 
        limited ressources, one image is divided into 4 parts, which ones passed into 
        the mask net. Only then we can run a classification task depending on the 
        different detected craters '''

        model = self.mask
        rdf_28, rdf_38, rdf_58 = self.rdf_28, self.rdf_38, self.rdf_58

        vgg_model_48 = vgg16.VGG16(weights='imagenet', input_shape=(48,48,3), include_top=False)
        intermediate_layer_model_48 = Model(inputs=vgg_model_48.input,
                                     outputs=vgg_model_48.layers[-1].output)

        vgg_model_58 = vgg16.VGG16(weights='imagenet', input_shape=(58,58,3), include_top=False)
        intermediate_layer_model_58 = Model(inputs=vgg_model_58.input,
                                         outputs=vgg_model_58.layers[-1].output)

        nb = X.shape[0]
        #fig, axes = plt.subplots(nrows=nb, ncols=2, figsize=(10, 180))
        
        y_pred_array = np.empty(X.shape[0], dtype=list)
        y_pred = np.empty(nb, dtype=list)

        ind = 0
        batch_size = 1
        
        def generator(X_, batch_size=batch_size):
            
            l_predict = []
            for j in range(X_.shape[0]):
                l_predict.extend([(X_[j][112:,112:]).reshape(1,112,112), (X_[j][:112,112:]).reshape(1,112,112),
                                  (X_[j][112:,:112]).reshape(1,112,112), (X_[j][:112,:112]).reshape(1,112,112)])
                if j == (X_.shape[0]-1):
                    yield np.concatenate(l_predict, axis=0).reshape((len(l_predict), 112, 112, 1))
                    break
                elif j%batch_size == 0 and j>0:
                    yield np.concatenate(l_predict, axis=0).reshape((len(l_predict), 112, 112, 1))
                    l_predict = []
                else : 
                    pass
        
        for X_batch in generator(X,batch_size):
            if ind == X.shape[0] :
                break
            X_pred = np.zeros((int(X_batch.shape[0]//4),224,224))
            pred = model.predict(X_batch)
            pred = pred.reshape((X_batch.shape[0],112,112))
            for i in range(0,pred.shape[0], 4) :
                X_pred[int(i//4), 112:,112:] = pred[i]
                X_pred[int(i//4), :112,112:] = pred[i+1]
                X_pred[int(i//4), 112:,:112] = pred[i+2]
                X_pred[int(i//4), :112,:112] = pred[i+3]

            for i in range(X_pred.shape[0]):

                image =np.ceil(X_pred[i]/255)*255
                edges = canny(image, sigma=3, low_threshold=10, high_threshold=50)
    
                # Detect two radii
                hough_radii = np.arange(4, 31, 1)
                hough_res = hough_circle(edges, hough_radii)
    
                # Select the most prominent 5 circles
                accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=20)
    
                circle = [(0.99, cx[k], cy[k], radii[k]) for k in range(len(cx)) if accums[k] >=0.4]
                circle = NMS_size(circle)
                
                y_pred_array[ind+i] = circle
            ind += X_pred.shape[0]

    
        print(y_pred)
        y_pred = np.empty(nb, dtype=list)

        b28, b38, b48, b58, b68 = [[],[]], [[],[]], [[],[]], [[],[]], [[],[]]
        indices_boxes = []
        for i in range(X.shape[0]):
            b28_, b38_, b48_, b58_, b68_ = get_boxes_from_circle(X[i], y_pred_array[i])
            b28 = [b28[0] + b28_[0], b28[1] + b28_[1]]
            b38 = [b38[0] + b38_[0], b38[1] + b38_[1]] 
            b48 = [b48[0] + b48_[0], b48[1] + b48_[1]] 
            b58 = [b58[0] + b58_[0], b58[1] + b58_[1]]
            b68 = [b68[0] + b68_[0], b68[1] + b68_[1]]
            
            indices_boxes.append([len(b28_[1]), len(b38_[1]), len(b48_[1]), len(b58_[1]), len(b68_[1]) ])
    
        if len(b28[0]) != 0 :
            
            intermediate_output = intermediate_layer_model_48.predict(np.array(b28[0]).reshape((len(b28[0]),48,48,3)))
            #print(type(intermediate_output))
            #print(len(b28[0]), len(intermediate_output))
            pred_list = rdf_28.predict_proba(np.array(intermediate_output).reshape(len(intermediate_output),512))
            
            s = 0
            for i in range(len(indices_boxes)):
                l = y_pred[i]
                if l == None : 
                    l = []
                    
                for j in range(0,indices_boxes[i][0]):
                    pred = pred_list[s+j]
                    acc, x, y, r = b28[1][s+j]
                    l.append((pred[1], x, y, r))
                y_pred[i] = l   
                s+= indices_boxes[i][0]
            
        if len(b38[0]) != 0 :
            intermediate_output = intermediate_layer_model_48.predict(np.array(b38[0]).reshape((len(b38[0]),48,48,3)))
                
            s = 0
            for i in range(len(indices_boxes)):
                l = y_pred[i]
                for j in range(0,indices_boxes[i][1]):
                    pred = rdf_38.predict_proba(intermediate_output[s+j].reshape((1,512)))
                    acc, x, y, r = b38[1][s+j]
                    l.append((pred[0][1], x, y, r))
                y_pred[i] = l   
                s+= indices_boxes[i][1]
            
        if len(b48[0]) != 0 :
            l_b48 = []
            for i in range(len(b48[0])):
                if b48[0][i].shape == (58, 58, 3):
                    l_b48.append(b48[0][i])
                else : 
                    l_b48.append(np.zeros((58, 58, 3)))
                    
            intermediate_output = intermediate_layer_model_58.predict(np.array(l_b48).reshape((len(l_b48),58,58,3)))
        
            s = 0
            for i in range(len(indices_boxes)):
                l = y_pred[i]
                for j in range(0,indices_boxes[i][2]):
                    pred = rdf_58.predict_proba(intermediate_output[s+j].reshape((1,512)))
                    acc, x, y, r = b48[1][s+j]
                    l.append((pred[0][1], x, y, r))              
                y_pred[i] = l   
                s+= indices_boxes[i][2]

        if len(b58[0]) != 0 :
            l_b58 = []
            for i in range(len(b58[0])):
                if b58[0][i].shape == (58, 58, 3):
                    l_b58.append(b58[0][i])
                else : 
                    l_b58.append(np.zeros((58, 58, 3)))

            intermediate_output = intermediate_layer_model_58.predict(np.array(l_b58).reshape((len(l_b58),58,58,3)))
        
            s = 0
            for i in range(len(indices_boxes)):
                l = y_pred[i]
                for j in range(0,indices_boxes[i][3]):
                    pred = rdf_58.predict_proba(intermediate_output[s+j].reshape((1,512)))
                    acc, x, y, r = b58[1][s+j]
                    l.append((pred[0][1], x, y, r))
                y_pred[i] = l   
                s+= indices_boxes[i][3]
        
        if len(b68[0]) != 0 :
            l_b68 = []
            for i in range(len(b68[0])):
                if b68[0][i].shape == (58, 58, 3):
                    l_b68.append(b68[0][i])
                else : 
                    print(b68[0][i].shape)
                    l_b68.append(np.zeros((58, 58, 3)))
            
            intermediate_output = intermediate_layer_model_58.predict(np.array(l_b68).reshape((len(l_b68),58,58,3)))
        
            s = 0
            for i in range(len(indices_boxes)):
                l = y_pred[i]
                for j in range(0,indices_boxes[i][4]):
                    pred = rdf_58.predict_proba(intermediate_output[s+j].reshape((1,512)))
                    acc, x, y, r = b68[1][s+j]
                    l.append((pred[0][1], x, y, r))
                y_pred[i] = l   
                s+= indices_boxes[i][4]    
            
        return y_pred 
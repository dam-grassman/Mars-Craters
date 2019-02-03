# -*- coding: utf-8 -*-
"""
Some functions that will be useful for many tasks, especially pre-processing of crater images  
"""

import random
import numpy as np 
from PIL import Image
from math import sqrt
import matplotlib.pyplot as plt

def reshape_image(array, basewidth=300, deepth =1):
    
    ''' Reshape a squared image with a given basewidth, 
    and a given deepth - usually 1 or 3 '''

    img = Image.fromarray(np.array(array))
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize))
    data = np.asarray(img)
    if deepth == 3 : 
        data = data.reshape(((basewidth, hsize,1)))
        data = np.concatenate([data, data, data],axis=2)
    return data

def get_crater(image, label, default_radius = None):
    
    ''' Get small crater-focused images centered on all the craters 
    of the image. Default sizes are 28, 38, 48, 58, 68. Default radius if 
    the same size is required whatever the crater radius is'''
    
    craters_images = []
    for x,y,r in label :
        if default_radius == None :
            r = (r - r%5 + 9)
        else :
            r = default_radius
        img = image[int(x-r):int(x+r), int(y-r):int(y+r)]
        if img.shape[0]==img.shape[1] and img.shape[0] != 0:
            craters_images.append(img) 
    return craters_images

def get_dictionnary_craters(array_x, array_y, default_radius=None):
    
    ''' Get the dictionnary of the craters-images library by size :
        Can be either 28, 38, 48, 58, 68'''
        
    dic_size = {}
    for ind in range(array_x.shape[0]):
        list_craters = get_crater(array_x[ind], array_y[ind], default_radius)
        for image in list_craters:
            l = dic_size.get(image.shape[0], [])
            l.append(image)
            dic_size[image.shape[0]] = l
            
    return dic_size


def dist(x1,y1,x2,y2):
    
    ''' Euclidean distance between to points on an image '''
    
    return sqrt((x1-x2)**2+(y1-y2)**2)

def construct_mask(list_circle):
    
    ''' Construct the mask of an image containing craters. 
    Given a list of craters, we return a image full of 0 ie black with 
    white circle (1) around craters '''
    
    shape = np.zeros((224,224))
    for circle in list_circle:
        x,y,r = circle
        for a in range(max(0,int(x-r)), min(int(x+r), 224)):
            for b in range(max(0,int(y-r)), min(224,int(y+r))):
                if dist(a,b,x,y) <= r:
                    shape[a,b] = 1
    return shape

def divide_image(img, label):
    
    '''Given an image and a list of craters, return a sub-list
    of (112,112) images (1/4 of an original image), only if 
    containing at least one craters (ie np.max = 1) '''
    
    l = []
    mask = construct_mask(label)
    if np.max(mask[:112,:112]) == 1 and np.sum(mask[:112,:112]) >= 45 : 
        
        l.append((img[:112,:112]/255, mask[:112,:112]))
    if np.max(mask[112:,:112]) == 1 and np.sum(mask[112:,:112]) >= 45 : 
        l.append((img[112:,:112]/255, mask[112:,:112]))
    if np.max(mask[:112,112:]) == 1 and np.sum(mask[:112,112:]) >= 45 : 
        l.append((img[:112,112:]/255, mask[:112,112:]))
    if np.max(mask[112:,112:]) == 1 and np.sum(mask[112:,112:]) >= 45 : 
        l.append((img[112:,112:]/255, mask[112:,112:]))
    return l 

def construct_circle_plan(circle, plot = False):
    
    ''' Similar to construct mask, it takes here only 1 circle '''
    
    shape = np.zeros((224,224))
    x,y,r = circle
    for a in range(max(0,int(x-r)), min(int(x+r), 224)):
        for b in range(max(0,int(y-r)), min(224,int(y+r))):
            shape[a,b] = 255
    if plot :
        fig, ax = plt.subplots()
        ax.imshow(shape, cmap='Greys_r')
    return shape

def get_image_augmented(img, label):
    
    ''' Construct an "augmented" image, wich is basically a deep image 
    (max 5 circle) with the true image, following by 4 different mask, 
    containing only one crater at a time '''
        
    img2 = img.reshape(224,224,1).copy()
    for circle in label : 
        #print(circle)
        shape = construct_circle_plan(circle, False)
        img2 = np.concatenate([img2, shape.reshape(224,224,1)], axis=2)
    while img2.shape[2] < 5 :
        shape =np.zeros((224,224))
        img2 = np.concatenate([img2, shape.reshape(224,224,1)], axis=2)
    return img2

def get_circle_from_array(shape):
    
    ''' Get coordinate and radius of a crater given a mask (array)
    with only one crater '''
    
    shape = reshape_image(shape, 224)
    max_ = np.max(shape)
    coord = np.argwhere(shape == max_)
    
    x_min = np.min(coord[:,1])
    x_max = np.max(coord[:,1])
    y_min = np.min(coord[:,0])
    y_max = np.max(coord[:,0])

    x, y = (x_max + x_min)/2 ,  (y_max + y_min) /2 
    
    if max(x_max-x_min,y_max - y_min)/min(x_max-x_min,y_max - y_min) > 1.3:
        return 111.5,111.5, 111.5
    radius = sqrt(max(x_max-x,y_max - y)**2)
    return y,x, radius

def fix_labels(image):
    
    ''' Fix the list of craters. If it gets a stange one (ie 111.5), 
    it won't select it as a label '''
    
    nb_circles = image.shape[2]
    new_list = []
    for i in range(1,nb_circles):
        img_2 = reshape_image(image[::,::,i]).reshape((300,300))
        x,y,r = get_circle_from_array(img_2)
        if r == 111.5:
            pass
        else :
            new_list.append((x,y,r))
    return new_list

def verify(liste):
    
    ''' Verify that the radius is at least 12 '''
    
    for x,y,r in liste : 
        if r > 12 : 
            return True 
    return False

def merge_dic(dic1,dic2):
    
    ''' Merge dictionnary of lists, with same keys. It will basically
    extend the list of corresponding keys.'''
    
    for key in dic1.keys() :
        if len(dic1[key]) < 6000 :
            l = dic1[key]
            l.extend(dic2.get(key,[]))
            dic1[key] = l
    return dic1


def get_random_box(img, size):
    
    ''' Get random box of given size on a image '''
    
    x = random.randint(size/2,img.shape[0]-size/2)
    y = random.randint(size/2,img.shape[0]-size/2)
    box = img[int(x-size/2):int(x+size/2), int(y-size/2):int(y+size/2)]
    return box

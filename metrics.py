# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 13:23:27 2019

@author: damie
"""

def iou(box1, box2):
    
    """Implement the intersection over union (IoU) between box1 and box2
    
    Arguments:
    box1 -- first box, list object with coordinates (x1, y1, x2, y2)
    box2 -- second box, list object with coordinates (x1, y1, x2, y2)
    """

    # Calculate the (y1, x1, y2, x2) coordinates of the intersection of box1 and box2. Calculate its Area.
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])

    if (xi2 - xi1) < 0 or (yi2 - yi1)<0 : 
        return -1
    inter_area = (xi2 - xi1) * (yi2 - yi1)


    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    
    # compute the IoU
    iou = inter_area / union_area
    
    return iou

def NMS(pred_list, threshold = 0.2):
    new_list_pred = []
    list_pred = sorted(pred_list, key=lambda row : row[2], reverse=True)
    while len(list_pred)!=0:
        #print(len(list_pred))
        largest_box = list_pred.pop(0)
        new_list_pred.append(largest_box)
        box_to_remove = []
        for i,box in enumerate(list_pred[::-1]) : 
            iou_ = iou([largest_box[0]-largest_box[2], largest_box[1]-largest_box[2], 
                        largest_box[0]+largest_box[2], largest_box[1]+largest_box[2]], 
                      [box[0]-box[2], box[1]-box[2], 
                       box[0]+box[2], box[1]+box[2]])

            if iou_>= threshold : 
                #print('DELETED')
                #print(len(list_pred)-1-i)
                box_to_remove.append(len(list_pred)-1-i)
        for ind in box_to_remove : 
            list_pred.pop(ind)
    #print(len(new_list_pred))
    return new_list_pred



def NMS_size(pred_list, threshold = 0.0):
    
    ''' Non-Suppression Max sorted by size and confidence'''
    
    new_list_pred = []
    if len(pred_list) == 0 :
        return []
    list_pred = sorted(sorted(pred_list, key=lambda row : row[0], reverse=True), key=lambda row : row[3], reverse=True)
    while len(list_pred)!=0:
        #print(len(list_pred))
        largest_box = list_pred.pop(0)
        new_list_pred.append(largest_box)
        box_to_remove = []
        for i,box in enumerate(list_pred[::-1]) : 
            iou_ = iou([largest_box[1]-largest_box[3], largest_box[2]-largest_box[3], 
                        largest_box[1]+largest_box[3], largest_box[2]+largest_box[3]], 
                      [box[1]-box[3], box[2]-box[3], 
                       box[1]+box[3], box[2]+box[3]])
            if iou_>= threshold : 

                box_to_remove.append(len(list_pred)-1-i)
        for ind in box_to_remove : 
            list_pred.pop(ind)

    return new_list_pred
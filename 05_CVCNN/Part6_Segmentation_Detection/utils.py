#!/usr/bin/env python
# coding: utf-8

# In[4]:


import matplotlib.pyplot as plt
import numpy as np
import albumentations as A
from albumentations import DualTransform
import tensorflow as tf
import cv2
import os
import random
import time


# In[69]:


class patience:    
    def __init__(self, patience=None):
        self.patience = patience
        self.record_value = 0
        self.early_stop = False
        self.time_dic = {}
        self.ini = True
        self.epoch_record = 0
            
    def record(self):
        if self.patience != -1:
            if self.ini and not self.patience:
                print('No early stop')
                self.ini = False
                self.patience = -1
            else:
                if self.record_value >= self.patience:
                    print('early stop')
                    self.early_stop = True
                self.record_value += 1
    def reset(self):
        self.record_value = 0
        
    def time(self, times_to_show=50, show=False):
#         assert self.time_dic, '初始化時未開始計時'
#         if not self.time_dic:
#             self.time_dic = {}
        epoch = self.epoch_record 
        if len(self.time_dic) == 0:
            key = 0
        else:
            key = list(self.time_dic.keys())[-1]+1
        self.time_dic[key] = time.time()
        
        if show:

            if (epoch)%(times_to_show) == 0:
                dic = self.time_dic
                lis = ['{0:.3f}'.format(dic[i*times_to_show]-dic[0]) for i in range(epoch//times_to_show+1)]
#                 print(list(range(epoch//times_to_show+1)))
                print(f'{self.epoch_record} epochs passed, ptime monitor every {times_to_show} epoch {lis}')
                

        self.epoch_record +=1
        return self.time_dic

        


# In[70]:


if __name__ == '__main__':
    monitor = patience(None)
    for i in range(151):
        monitor.time(show=True)
#     pass


# In[59]:


def show_image_mask(*img_, split=False):
    plt.figure(figsize=(5,100))
    for i, img in enumerate(list(img_), 1):
        plt.subplot(1,len(img_),i)
        if len(img.shape)==4:
            img = tf.reshape(img, [img.shape[0]*img.shape[1], img.shape[2], img.shape[3]])
            if img.shape[-1] != 3:
                img = tf.squeeze(img)
        
        if len(np.shape(img)) == 2:
            plt.imshow(np.array(img).astype(np.int32), cmap='gray')
        else:
            plt.imshow(np.array(img).astype(np.int32))
    plt.show()
    plt.close()
    


# In[60]:


def find_objects_contours(mask):
    thresh = mask
    contours, hier =         cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    np.shape(contours)

    arr = np.array(contours)[-1].reshape(-1,2)
    arr = arr.mean(axis=0)
    return arr


# In[61]:


if __name__ == '__main__':
    pass


# In[62]:


def center_to_4point(mask, arr, side_width, pad=25):
    limit = len(mask)
    points = [0]*4
    if not pad:
        pad = 0
    value = side_width/2+pad    
    for i in arr:
        if side_width+2*pad > limit:
            print(side_width+2*pad)
            raise ValueError('not enough')
        if i > limit:
            raise ValueError('not include')
            
    for i in range(len(points)):
        if i in [0,1]:
            if arr[i%2] - value < 0:
                points[i] = 0
                points[i+2] += np.abs(arr[i%2] - value)
            else:
                points[i] = arr[i%2]-value
        if i in [2,3]:
            if arr[i%2]+value > limit:
                print(arr[i%2]+value)
                points[i] = len(mask)
                points[i-2] -= np.abs(limit - arr[i%2] - value)
            else:
                points[i] = arr[i%2]+value
    
    return np.round(points).astype(int)


# In[63]:


class mask_CutMix(DualTransform):
    def __init__(self,img_mask_list, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.img_lis, self.mask_lis = zip(*img_mask_list)
        self.choice = np.random.choice(range(len(self.img_lis)),size=1, replace=False)
        self.seed = 1000
        
    def apply(self, img, **params):
        a = self.choice[0]
#         a = choice[0]
#         b = choice[1]
#         print(a,b)
        source_center = self.find_objects_contours(self.mask_lis[a])
        points, _ = self.center_to_4point(self.mask_lis[a], source_center, 256)
        
        target_image = img
        if len(np.shape(img)) == 2:
            source_image = self.mask_lis[a]
        else:
            source_image = self.img_lis[a]
            self.seed = np.random.choice(range(10000),size=1)[0]
        
    
        x_min, y_min, x_max, y_max = points
        target_image = target_image.copy()
        piece = source_image[y_min:y_max, x_min:x_max]
        
        
        transform = A.Compose([
                A.Rotate((-30, 30), p=1), 
                A.RandomBrightnessContrast(brightness_limit=[-0.05, 0.05], p=0.2),
                A.HorizontalFlip(p=0.5),
            ])
        random.seed(self.seed)
        transformed  = transform(image=piece)

        
        target_image[y_min:y_max, x_min:x_max] = transformed['image']
        return target_image
        
    def find_objects_contours(self,mask):
        thresh = mask
        contours, hier =             cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        np.shape(contours)

        arr = np.array(contours)[-1].reshape(-1,2)
        arr = arr.mean(axis=0)
        return arr

    def center_to_4point(self, mask, arr, side_width, pad=None):
        limit = len(mask)
        points = [0]*4

        if not pad:
            pad = 0
        value = side_width/2+pad    
        for i in arr:
            if side_width+2*pad > limit:
                print(side_width+2*pad)
                raise ValueError('not enough')
            if i > limit:
                raise ValueError('not include')

        for i in range(len(points)):
            if i in [0,1]:
                if arr[i%2] - value < 0:
                    points[i] = 0
                    points[i+2] += np.abs(arr[i%2] - value)
                else:
                    points[i] = arr[i%2]-value
            if i in [2,3]:
                if arr[i%2]+value > limit:
                    print(arr[i%2]+value)
                    points[i] = len(mask)
                    points[i-2] -= np.abs(limit - arr[i%2] - value)
                else:
                    points[i] = arr[i%2]+value
        points = np.round(points).astype(int) 
        x_min, y_min, x_max, y_max = points
        return points, mask[y_min:y_max, x_min:x_max]


# In[64]:


if __name__ == '__main__':
    if get_ipython().__class__.__name__ =='ZMQInteractiveShell':
        os.system('jupyter nbconvert utils.ipynb --to python')


# In[ ]:





# In[ ]:





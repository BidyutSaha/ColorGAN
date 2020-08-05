#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)


# In[2]:


#import loss

from numpy import loadtxt
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt


# In[4]:


import tensorflow.keras.backend as K


def  l2(y_true, y_pred):
   
    x=K.mean(K.square(y_true-y_pred),axis=-1)
    return x


def  l1(y_true, y_pred):
    
    x=K.mean(K.abs(y_true-y_pred),axis=-1)
    return x




def  l3(y_true, y_pred):
    
    x=K.mean(K.sqrt(K.sum(K.square(y_true-y_pred),axis=-1)))
    return x


# In[5]:


custom_losses = {
    "l1" : l1,
    "l2" : l2,
    "l3":l3
    
}


# In[6]:


def histIntersection_1D(a,b,x,y,bins):
  a = a.ravel()
  b = b.ravel()
  bins=np.linspace(x,y,bins)
  
  x,_ = np.histogram(a,bins=bins,range=(x,y))
  x= x/x.sum()
 

  y,_ = np.histogram(b,bins=bins,range=(x,y))
  y= y/y.sum()
 
 
  return np.minimum(x,y).sum()


def histogramIntersection_2d(a,b,x,y,bins):
  alpha_a = a[...,0].ravel()
  
  beta_a = a[...,1].ravel()
  a_2dhist,_,_ = np.histogram2d(alpha_a,beta_a,bins=np.linspace(x,y,bins))

  a_2dhist = a_2dhist/a_2dhist.sum()
  #print(a_2dhist)

  alpha_b = b[...,0].ravel()
  beta_b = b[...,1].ravel()
  b_2dhist,_,_ = np.histogram2d(alpha_b,beta_b,bins=np.linspace(x,y,bins))
  b_2dhist = b_2dhist/b_2dhist.sum()
  #print(b_2dhist)

  return np.minimum(a_2dhist,b_2dhist).sum()


# In[8]:



# load and prepare training images
import numpy as np
def load_real_samples(filename):
    data = np.load(filename)
    X1, X2 = data['arr_0'], data['arr_1']
    return [X1, X2]


# In[9]:


...
# load image data
from numpy.random import randint
from numpy import ones,zeros
dataset = load_real_samples('../Dataset/train.npz')
testset = load_real_samples('../Dataset/test.npz')
valset = load_real_samples('../Dataset/val.npz')


# In[16]:


[src_images_test, tar_images_test]= dataset
[src_images_val, tar_images_val]= testset
[src_images_train, tar_images_train]= valset

[src_images_test, tar_images_test]= load_real_samples('../Dataset/finaltest.npz')
[src_images_val, tar_images_val]= load_real_samples('../Dataset/linn256val.npz')
[src_images_train, tar_images_train]= load_real_samples('../Dataset/linn256train.npz')
# In[7]:


# load, split and scale the maps dataset ready for training
from os import listdir
from numpy import asarray
from numpy import vstack
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from numpy import savez_compressed
from skimage import color,io
import matplotlib.pyplot as plt
import numpy as np


# load all images in a directory into memory
def load_images(path):
  gray_list, ab_list = list(), list()
	# enumerate filenames in directory, assume all are images
  for filename in listdir(path):
		# load  the image
    rgb = io.imread(path + filename)
    
    #plt.subplot(1,2,1)
    #plt.imshow(rgb)

    lab = color.rgb2lab(rgb)

    
    scalled_lab = lab/100

    #reconstructed_lab = scalled_lab * 100
    #generated_rgb = color.lab2rgb(reconstructed_lab)
    #plt.subplot(1,2,2)

    #plt.imshow(generated_rgb)

    #print(pixels.shape)

    # split into gray and ab
    gray, ab = scalled_lab[:,:,0].reshape((256,256,1)), scalled_lab[:, :,1:]
    gray_list.append(gray)
    ab_list.append(ab)
  return [asarray(gray_list), asarray(ab_list)]


# In[10]:


model_files = ["../SavedModels/_gen_model1a.h5","../SavedModels/_gen_model1b.h5","../SavedModels/_gen_model2a.h5","../SavedModels/_gen_model2b.h5"]
models= []


# In[11]:


def load_models(dir=""):
 
    for m in model_files:
        fpath = dir+m
        model = load_model(fpath,custom_objects = custom_losses)
        models.append(model)


# In[12]:


def getSummary(datasetType=1,model=None):
    tar_images = None
    src_images = None
    pred_images = None
    if datasetType == 1 :
        src_images = src_images_train
        tar_images = tar_images_train
    elif datasetType == 2 :
        src_images = src_images_test
        tar_images = tar_images_test
    else :
        src_images = src_images_val
        tar_images = tar_images_val
        
    if model != None :
        pred_images = model.predict(src_images)
        x=-1
        y=1
        bins=200
        print("AB" , histogramIntersection_2d(tar_images,pred_images,x,y,bins))
        print("A",histIntersection_1D(tar_images[...,0],pred_images[...,0],x,y,bins))
        print("B",histIntersection_1D(tar_images[...,1],pred_images[...,1],x,y,bins))
        print("=======================")
        return
        
        
    for model in models :
        pred_images = model.predict(src_images)
        x=-1
        y=1
        bins=200
        print("AB" , histogramIntersection_2d(tar_images,pred_images,x,y,bins))
        print("A",histIntersection_1D(tar_images[...,0],pred_images[...,0],x,y,bins))
        print("B",histIntersection_1D(tar_images[...,1],pred_images[...,1],x,y,bins))
        print("=======================")


# In[13]:


models= []
load_models()
len(models)


# In[14]:


models


# In[1]:


#train
#getSummary(1)


# In[2]:


#model = load_model("../outputs/generator_finalgl3model1_after50.h5",custom_objects = custom_losses)
#models.append(model)


# In[3]:


#test
#getSummary(2)


# In[4]:


#valid
#getSummary(3)

getSummary(1,models[5])getSummary(1,models[3])getSummary(2,models[2])
# In[21]:


from skimage  import color

def getRGB(gray,ab):
  l = gray.reshape((256,256)) * 100
  a = ab[:,:,0].reshape((256,256)) * 100
  b = ab[:,:,1].reshape((256,256)) * 100
  rgb = color.lab2rgb(np.dstack((l,a,b)))
  return rgb


# In[19]:


def imgPlot(indices,model,datasetType=1):
    emptyCanvas=True
    canvas=None
    tar_images = None
    src_images = None
    pred_images = None
    if datasetType == 1 :
        src_images = src_images_train[indices]
        tar_images = tar_images_train[indices]
    elif datasetType == 2 :
        src_images = src_images_test[indices]
        tar_images = tar_images_test[indices]
    else :
        src_images = src_images_val[indices]
        tar_images = tar_images_val[indices]
        
    pred_images = model.predict(src_images)
    for imgid in range(len(indices)):
        x=src_images[imgid].reshape((256,256))
        imggray=np.dstack((x,x,x))
        imgpred=getRGB(src_images[imgid],pred_images[imgid])
        imgtrue=getRGB(src_images[imgid],tar_images[imgid])
        #plt.figure(figsize=(20,10))
        #plt.imshow(img)
        img=np.hstack((imggray,imgtrue,imgpred))
        
        
        #return 
        if emptyCanvas == True :
            canvas=img
            emptyCanvas = False
            
        else:
            canvas=np.vstack((canvas,img))
            
    #plt.imshow(canvas)
    return canvas



def imgPlot_single(indices,model,datasetType=1):
    emptyCanvas=True
    canvas=None
    tar_images = None
    src_images = None
    pred_images = None
    if datasetType == 1 :
        src_images = src_images_train[indices]
        tar_images = tar_images_train[indices]
    elif datasetType == 2 :
        src_images = src_images_test[indices]
        tar_images = tar_images_test[indices]
    else :
        src_images = src_images_val[indices]
        tar_images = tar_images_val[indices]
        
    pred_images = model.predict(src_images)
    for imgid in range(len(indices)):
     
        imgpred=getRGB(src_images[imgid],pred_images[imgid])
       
        img=imgpred
        
        
        #return 
        if emptyCanvas == True :
            canvas=img
            emptyCanvas = False
            
        else:
            canvas=np.vstack((canvas,img))
            
    #plt.imshow(canvas)
    return canvas


# In[35]:


N=12
plt.figure(figsize=(20,20))
indices=np.random.randint(0,50,size=N)
print(indices)
dtype=1
a=imgPlot(indices,models[0],dtype)
#plt.imshow(a)
#############second#########################
# b=imgPlot_single(indices,models[1],2)
# a= np.hstack((a,b))
for i in range(1,len(models)):
    b=imgPlot_single(indices,models[1],dtype)
    a= np.hstack((a,b))
plt.figure(figsize=(20,20))
plt.imshow(a) 


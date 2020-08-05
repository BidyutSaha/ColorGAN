#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
#from tensorflow.keras.models import Input
from tensorflow.keras.layers import Conv2D,Input,Lambda
from tensorflow.keras.layers import Conv2DTranspose,UpSampling2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.utils import plot_model
import tensorflow.keras.backend as K
import tensorflow





def replct(x):
    return tensorflow.concat((x,x), axis=-1)


import numpy as np
def l3(y_true, y_pred):
    x= (K.mean(K.sqrt(K.sum(K.square(y_true-y_pred),axis=-1))))
    return x


# define the discriminator model
def define_discriminator(condition_shape,target_shape):
    # weight initialization
    init =  RandomNormal(stddev=0.01)
   
    # target image input
    in_target_image = Input(shape=target_shape)
    
     # source image input
    in_cnd_image = Input(shape=condition_shape)
    in_cnd_image_=Lambda(replct)(in_cnd_image)
   
    # concatenate images channel-wise
    merged = Concatenate(axis=-1)([ in_cnd_image_,in_target_image])
    # C64
    d = Conv2D(64, (4,4), strides=(2,2), padding='same')(merged)
    d = LeakyReLU(alpha=0.2)(d)
    # C128
    d = Conv2D(128, (4,4), strides=(2,2), padding='same')(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C256
    d = Conv2D(256, (4,4), strides=(2,2), padding='same')(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # C512
    d = Conv2D(512, (4,4), strides=(2,2), padding='same')(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # second last output layer
    d = Conv2D(512, (4,4), padding='same')(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    # patch output
    d = Conv2D(1, (4,4), padding='same')(d)
    patch_out = Activation('sigmoid')(d)
    # define model
    model = Model([in_cnd_image, in_target_image], patch_out)
    # compile model
    opt = Adam(lr=0.00001, beta_1=0.9)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[1])
    return model


# define image shape
image_shape = (256,256,1)
condition_shape = (256,256,1)
target_shape = (256,256,2)



# define the models
d_model = define_discriminator(condition_shape,target_shape)


# In[2]:


# summarize the model
# d_model.summary()
# plot the model
#plot_model(d_model, show_shapes=True, show_layer_names=True)


# In[3]:



# define an encoder block
def define_encoder_block(layer_in, n_filters, batchnorm=True):
    # weight initialization
   
    # add downsampling layer
    g = Conv2D(n_filters, (3,3), strides=(2,2), padding='same')(layer_in)
    # conditionally add batch normalization
    if batchnorm:
        g = BatchNormalization()(g)
    # leaky relu activation
    g = LeakyReLU(alpha=0.2)(g)
    return g

# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
    # weight initialization

    # add upsampling layer
    #g = Conv2DTranspose(n_filters, (3,3), strides=(2,2), padding='same')(layer_in)
    g= UpSampling2D()(layer_in)
    g = Conv2D(n_filters, (3,3), strides=(1,1), padding='same')(g)

    # add batch normalization
    g = BatchNormalization()(g)
    # conditionally add dropout
    if dropout:
        g = Dropout(0.5)(g)
    # merge with skip connection
    g = Concatenate()([g, skip_in])
    # relu activation
    g = Activation('relu')(g)
    return g

# define the standalone generator model
def define_generator(image_shape=(256,256,1)):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=image_shape)
    # encoder model: C64-C128-C256-C512-C512-C512-C512-C512
    e1 = define_encoder_block(in_image, 64, batchnorm=False)
    e2 = define_encoder_block(e1, 128)
    e3 = define_encoder_block(e2, 256)
    e4 = define_encoder_block(e3, 512)
    e5 = define_encoder_block(e4, 512)
    e6 = define_encoder_block(e5, 512)
    e7 = define_encoder_block(e6, 512)
    # bottleneck, no batch norm and relu
    b = Conv2D(512, (2,2), strides=(2,2), padding='same',kernel_initializer=init)(e7)
    b = Activation('relu')(b)
    # decoder model: CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
    d1 = decoder_block(b, e7, 512)
    d2 = decoder_block(d1, e6, 512)
    d3 = decoder_block(d2, e5, 512)
    d4 = decoder_block(d3, e4, 512)
    d5 = decoder_block(d4, e3, 256)
    d6 = decoder_block(d5, e2, 128, dropout=False)
    d7 = decoder_block(d6, e1, 64, dropout=False)
    # output
    g = Conv2DTranspose(2, (3,3), strides=(2,2), padding='same',kernel_initializer=init)(d7)
    out_image = Activation('tanh')(g)
    # define model
    model = Model(in_image, out_image)
    return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, image_shape,condition_shape):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# define the source image
	in_src = Input(shape=image_shape)
	# connect the source image to the generator input
	gen_out = g_model(in_src)
	# connect the source input and generator output to the discriminator input
	dis_out = d_model([in_src, gen_out])
	# src image as input, generated image and classification output
	model = Model(in_src, [dis_out, gen_out])
	# compile model
	opt = Adam(lr=0.002, beta_1=0.5)
	model.compile(loss=['binary_crossentropy',l3], optimizer=opt, loss_weights=[1,0])
	return model


g_model = define_generator(image_shape)
# define the composite model
gan_model = define_gan(g_model, d_model, image_shape,target_shape)


# In[4]:



from os import listdir
from numpy import asarray
from skimage import color,io

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


# In[5]:


from numpy import savez_compressed


# In[6]:





# load  test dataset
train_set=[src_images_train, tar_images_train] = load_images(r'../Dataset/train/')
print('Loaded: ', src_images_train.shape, tar_images_train.shape)
# save as compressed numpy array
#filename = '../Dataset/linn256val.npz'
#savez_compressed(filename, src_images, tar_images)
#print('Saved dataset: ', filename)


# In[7]:


# save as compressed numpy array
filename = '../Dataset/train.npz'
savez_compressed(filename, src_images_train, tar_images_train)
print('Saved dataset: ', filename)


# In[8]:



# load  test dataset
test_set = [src_images_test, tar_images_test] = load_images(r'../Dataset/test/')
print('Loaded: ', src_images_test.shape, tar_images_test.shape)
# save as compressed numpy array
filename = '../Dataset/test.npz'
savez_compressed(filename, src_images_test, tar_images_test)
print('Saved dataset: ', filename)


# In[9]:



# load  val dataset
test_set = [src_images_val, tar_images_val] = load_images(r'../Dataset/val/')
print('Loaded: ', src_images_val.shape, tar_images_val.shape)
# save as compressed numpy array
filename = '../Dataset/val.npz'
savez_compressed(filename, src_images_val, tar_images_val)
print('Saved dataset: ', filename)


# In[6]:



# load and prepare training images
import numpy as np
def load_real_samples(filename):
	# load compressed arrays
	data = np.load(filename)
	# unpack arrays
	X1, X2 = data['arr_0'], data['arr_1']
	# scale from [0,255] to [-1,1]
	#X1 = (X1 - 127.5) / 127.5
	#X2 = (X2 - 127.5) / 127.5
	return [X1, X2]


# In[7]:


# select a batch of random samples, returns images and target
from numpy.random import randint
from numpy import ones,zeros
def generate_real_samples(dataset, n_samples, patch_shape):
	# unpack dataset
	trainA, trainB = dataset

    
	# choose random instances
	ix = randint(0, trainA.shape[0], n_samples)
	# retrieve selected images
	X1, X2 = trainA[ix], trainB[ix]
	# generate 'real' class labels (1)
	y = ones((n_samples, patch_shape, patch_shape, 1))
	return [X1, X2], y


# In[8]:


# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
	# generate fake instance
	X = g_model.predict(samples)
	# create 'fake' class labels (0)
	y = zeros((len(X), patch_shape, patch_shape, 1))*0.9
	return X, y


# In[31]:


# train pix2pix models
hist= []
def train(d_model, g_model, gan_model, dataset, n_epochs=25, n_batch=24, n_patch=16):
    # unpack dataset
    trainA, trainB = dataset
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    real_loss=[]
    fake_loss=[]
    
    
  # manually enumerate epochs
    for i in range(n_steps):
        # select a batch of real samples
        [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
        # generate a batch of fake samples
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
        # update discriminator for real samples
        d_loss1 = d_model.train_on_batch([ X_realA,X_realB], y_real)
        # update discriminator for generated samples
        d_loss2 = d_model.train_on_batch([X_realA,X_fakeB], y_fake)
        # update the generator
        [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
        g_loss_, p_, q_ = gan_model.train_on_batch(X_realA, [y_real, X_realB])  
        [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)

        g_loss, p, q = gan_model.train_on_batch(X_realA, [y_real, X_realB])
       
     
        g_loss = (g_loss + g_loss_)/2
        print( g_loss, (p+p_)/2, (q+q_)/2)
        # summarize performance
        print('>epoch * batch %d,epoch [%.3f]  ============== d1[%.3f] d2[%.3f] g[%.3f]' % (i+1,(i+1)/bat_per_epo, d_loss1, d_loss2, g_loss))
        hist.append([ d_loss1, d_loss2, g_loss])
        # summarize model performance
        if (i+1) % (bat_per_epo * 1) == 0:
            #summarize_performance(i, g_model, dataset)
            piclog(i,g_model,dataset)


# In[30]:


from matplotlib import pyplot


def getRGB(gray,ab):
  l = gray.reshape((256,256)) * 100
  a = ab[:,:,0].reshape((256,256)) * 100
  b = ab[:,:,1].reshape((256,256)) * 100
  rgb = color.lab2rgb(np.dstack((l,a,b)))
  return rgb


def piclog(step,g_model,dataset,n_samples=6):
    summarize_performance(step,g_model,dataset,1,n_samples)
    summarize_performance(step,g_model,testset,2,n_samples)

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, dataset,typ, n_samples=6):
  folder = "../Outputs/ColorGanL3/"
	# select a sample of input images
  fname = ""
  if typ == 2 :
    fname = str(step)+"after100____gantest_model1.png"
  else:
    fname = str(step)+"after100____gantrain_model1.png"
 
  [X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
	# generate a batch of fake samples
  X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)

  k=7
  pyplot.figure(figsize=(16,18.9))
	# plot real source images
  for i in range(n_samples):
    pyplot.subplot(k, n_samples, 1 + i)
    pyplot.axis('off')
    pyplot.imshow(X_realA[i].reshape((256,256)),cmap="gray")
	# plot generated target image
  for i in range(n_samples):
    pyplot.subplot(k, n_samples, 1 + n_samples + i)
    pyplot.axis('off')
    pyplot.imshow(getRGB(X_realA[i],X_fakeB[i]))
	# plot real target image
  for i in range(n_samples):
    pyplot.subplot(k, n_samples, 1 + n_samples*2 + i)
    pyplot.axis('off')
    pyplot.imshow(getRGB(X_realA[i],X_realB[i]))
###########################################################################
  for i in range(n_samples):
    pyplot.subplot(k, n_samples, 1 + n_samples*3 + i)
    pyplot.axis('off')
    x=(X_realB[i][:,:,0].reshape((256,256))+1)/2
    pyplot.imshow(x,cmap="gray")
    
  for i in range(n_samples):
    pyplot.subplot(k, n_samples, 1 + n_samples*4 + i)
    pyplot.axis('off')
    x=(X_fakeB[i][:,:,0].reshape((256,256))+1)/2
    pyplot.imshow(x,cmap="gray")
    
    
  for i in range(n_samples):
    pyplot.subplot(k, n_samples, 1 + n_samples*5 + i)
    pyplot.axis('off')
    x=(X_realB[i][:,:,1].reshape((256,256))+1)/2
    pyplot.imshow(x,cmap="gray")
    
  for i in range(n_samples):
    pyplot.subplot(k, n_samples, 1 + n_samples*6 + i)
    pyplot.axis('off')
    x=(X_fakeB[i][:,:,1].reshape((256,256))+1)/2
    pyplot.imshow(x,cmap="gray")
    
    
	# save plot to file
  filename1 = folder + fname
  pyplot.savefig(filename1)
  pyplot.close()
	# save the generator model
  if typ == 1 :
    
      filename2 = folder + 'fcolorgan.h5' 
      g_model.save(filename2)
  #print('>Saved: %s and %s' % (filename1, filename2))


# In[11]:


...
# load image data
from numpy.random import randint
from numpy import ones,zeros
dataset = load_real_samples('../Dataset/train.npz')
testset = load_real_samples('../Dataset/test.npz')


# In[12]:


valset = load_real_samples('../Dataset/val.npz')


# In[35]:


piclog(0,g_model,dataset)


# In[34]:


# train model
#hist=[]
train(d_model, g_model, gan_model, dataset)


# In[15]:


file = "../SavedModels/generator_finalgl3model1_after100gan.h5"
g_model.save(file)


# In[16]:


file = "../SavedModels/discriminitor_finalgl3model1_after100gan.h5"
d_model.save(file)


# In[17]:


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


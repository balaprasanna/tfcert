#!/usr/bin/env python
# coding: utf-8

# ## Quiz
#
# Suppose that we have a 10x10 image with only one colour channel. We apply a single convolutional filter with kernel size 3x3, stride 1 and no zero padding (‘VALID’ padding), followed by a 2x2 pooling layer (with a default stride of 2 in both dimensions). What are the dimensions of the output?

# In[1]:


import tensorflow as tf


# In[4]:


from tensorflow.keras.layers import *


# In[5]:


import numpy as np


# In[48]:


shape= (1, 10,10,1 )
inp = np.random.random(shape)
inp = inp.astype('float32')


# In[49]:


# inp_t = tf.convert_to_tensor(
#     inp, dtype='float32', name='inp'
# )


# In[53]:


kernel_size = (3,3)
strides = (2,2)
padding = 'valid'


# In[52]:


l1 = Conv2D(1, kernel_size=kernel_size, strides=strides, padding=padding, name="l1")
l2 = MaxPool2D()
l2(l1(inp)).shape


# In[61]:


l1 = Conv2D(1, kernel_size=kernel_size, strides=(2,2), padding=padding, name="l1")
# l2 = MaxPool2D()
l1(inp).shape


# In[ ]:





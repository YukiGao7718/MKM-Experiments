#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import the dataset
# from keras.datasets import mnist
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels,cosine_similarity


# In[3]:


#data = pd.read_csv('mnist_test.csv',header = None).iloc[:5000,1:].to_numpy()
#print(data.shape)


# In[35]:


# parameter X: ndarray of shape (n_sample, n_feature)
# parameter n: n-th order of arc-cosine kernel and it's also the dimension of activation used to derive the
# kernel.
# return norm_matrix of shape (n_sample,n_sample): the 1st part of explicit expression of arc-cosine kernel
def CalNorm(X,Y,n):
    size_X = np.array(X).shape[0]
    size_Y = np.array(Y).shape[0]
    pi_matrix = np.ones((size_X,size_Y))*(1/np.pi)
    
    x = np.linalg.norm(X,axis=1)
    y = np.linalg.norm(Y,axis=1)
    x = np.tile(x,(1,size_Y)).reshape(size_X,size_Y)
    y = np.tile(y,(1,size_X)).reshape(size_X,size_Y)
    norm_single_matrix = np.multiply(x,y)   
    norm_matrix = np.multiply(x,y)
    if n == 0:
        return pi_matrix
    else:
        for i in range(n-1):
            norm_matrix = np.multiply(norm_matrix,norm_single_matrix)
        norm_matrix = np.multiply(pi_matrix,norm_matrix)
        return norm_matrix    


# In[2]:


# parameter X: ndarray of shape (n_sample, n_feature)
# return theta matrix of shape (n_sample,n_sample)
def CalTheta(X,Y):
    return np.arccos(np.around(cosine_similarity(X,Y),8))


# In[30]:


# parameter X: ndarray of shape (n_sample, n_feature)
# parameter n: n-th order of arc-cosine kernel and it's also the dimension of activation used to derive the
# kernel.
def Jn(X,Y,n):
    theta_matrix = CalTheta(X,Y)
    size_X = np.array(X).shape[0]
    size_Y = np.array(Y).shape[0]
    pi_matrix = np.ones((size_X,size_Y))*(np.pi)
    if n == 0:
        return pi_matrix-theta_matrix
    elif n == 1:
        return np.sin(theta_matrix) + np.multiply((pi_matrix-theta_matrix),np.cos(theta_matrix))
    elif n == 2:
        cos_sqr_matrix = np.multiply(np.cos(theta_matrix),np.cos(theta_matrix))
        ones = np.ones(cos_sqr_matrix.shape)
        return 3*np.multiply(np.sin(theta_matrix),np.cos(theta_matrix)) + np.multiply((pi_matrix-theta_matrix),(ones + 2*cos_sqr_matrix))
    else:
        print('The order is out of range!')
    


# In[33]:


# procedure: element-wise multiply the 4 parts together to get kernel matrix
# parameter X: data array of shape (n_sample, n_feature)
# parameter n: n-th order of arc-cosine kernel and it's also the dimension of activation used to derive the
# kernel.
# return arc_cos_kmatrix: the resulting kernel matrix
def ArcCosineKernel(X,Y,n):
    arc_cos_kmatrix = np.multiply(CalNorm(X,Y,n),Jn(X,Y,n))
#     arc_cos_kmatrix = np.multiply(np.multiply(np.multiply(CalNorm(X,n),
#                                 CalSin(X,n)),CalSinDeri(X,n)),CalPiSin(X,n))
    print(arc_cos_kmatrix.shape)
    return arc_cos_kmatrix  


# In[36]:


#X = [[1,0,0],
#     [0,1,0],
#     [0,0,1]]
#Y = [[1,0,0],
#     [1,1,2]]
#print(ArcCosineKernel(X,Y,1))


# In[6]:


# procedure: element-wise multiply the 4 parts together to get kernel matrix
# parameter X: data array of shape (n_sample, n_feature)
# parameter n: n-th order of arc-cosine kernel and it's also the dimension of activation used to derive the
# kernel.
# return arc_cos_kmatrix: the resulting kernel matrix
def ArcCosineKernel0(X,Y):
    arc_cos_kmatrix = np.multiply(CalNorm(X,Y,0),Jn(X,Y,0))
#     arc_cos_kmatrix = np.multiply(np.multiply(np.multiply(CalNorm(X,n),
#                                 CalSin(X,n)),CalSinDeri(X,n)),CalPiSin(X,n))
    print(arc_cos_kmatrix.shape)
    return arc_cos_kmatrix  


# In[1]:


# procedure: element-wise multiply the 4 parts together to get kernel matrix
# parameter X: data array of shape (n_sample, n_feature)
# parameter n: n-th order of arc-cosine kernel and it's also the dimension of activation used to derive the
# kernel.
# return arc_cos_kmatrix: the resulting kernel matrix
def ArcCosineKernel1(X,Y):
    arc_cos_kmatrix = np.multiply(CalNorm(X,Y,1),Jn(X,Y,1))
#     arc_cos_kmatrix = np.multiply(np.multiply(np.multiply(CalNorm(X,n),
#                                 CalSin(X,n)),CalSinDeri(X,n)),CalPiSin(X,n))
    print(arc_cos_kmatrix.shape)
    return arc_cos_kmatrix  


# In[4]:


# procedure: element-wise multiply the 4 parts together to get kernel matrix
# parameter X: data array of shape (n_sample, n_feature)
# parameter n: n-th order of arc-cosine kernel and it's also the dimension of activation used to derive the
# kernel.
# return arc_cos_kmatrix: the resulting kernel matrix
def ArcCosineKernel2(X,Y):
    arc_cos_kmatrix = np.multiply(CalNorm(X,Y,2),Jn(X,Y,2))
#     arc_cos_kmatrix = np.multiply(np.multiply(np.multiply(CalNorm(X,n),
#                                 CalSin(X,n)),CalSinDeri(X,n)),CalPiSin(X,n))
    print(arc_cos_kmatrix.shape)
    return arc_cos_kmatrix  


# In[11]:


# parameter layer: composition layers of arccosine kernel
# return New_x: the resulting kernel matrix
def ArcCosineKernelComp(X,Y,n,layer = 1):
    New_X,New_Y = X,Y
    for i in range(layer):
        temp_X = ArcCosineKernel(New_X,New_Y,n)
        New_X = temp_X
    return New_X


# In[12]:


# print(CalNorm(data,0))
# print(Jn(data,0))
# print(ArcCosineKernel(data,2))
#print(ArcCosineKernelComp(data,2,layer = 1))


# In[ ]:





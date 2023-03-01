#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import math
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# dimension reduction
def lda_dim_reduction(x_train, y_train, x_test, n_components):
    '''
    x_train: train feature set
    y_train: train image labels
    x_test: test feature set
    n_components: the number of components left after lda
    '''
    
    # Fit the LDA model on train features
    lda = LDA(n_components=n_components)
    lda.fit(x_train,y_train)
    
    # Project data to maximize class separation
    redx_train=lda.transform(x_train)
    redx_test=lda.transform(x_test)
    
    return redx_train,redx_test

def iris_match(redx_train, redx_test):
    '''
    redx_train: train feature set after lda reduction
    redx_test: test feature set after lda reduction
    '''
    
    index_d1 = []
    index_d2 = []
    index_cos = []
    roc_value = []
    
    
    for i in range(len(redx_test)):
        d1=[]
        d2=[]
        d3=[]

        # compare ith test image with every train image
        for j in range(len(redx_train)):
            f=redx_test[i]
            fi=redx_train[j]
            L1=0
            L2=0
            fsq_sum=0
            fisq_sum=0
            cos_dist=0

            # L1 and L2 distance and sum of squares of f and fi
            for k in range(0,len(f)):
                L1+=abs(f[k]-fi[k])
                L2+=(f[k]-fi[k])**2
                fsq_sum += f[k]**2
                fisq_sum += fi[k]**2

            # cosine distance
            cos_dist=1-((np.matmul(f,np.transpose(fi)))/(math.sqrt(fsq_sum)*math.sqrt(fisq_sum)))

            d1.append(L1)
            d2.append(L2)
            d3.append(cos_dist)

        # Store the minimum distance index and minimum distance value for roc plot
        index_d1.append(d1.index(min(d1)))
        index_d2.append(d2.index(min(d2)))
        index_cos.append(d3.index(min(d3)))
        roc_value.append(min(d3))
    return index_d1, index_d2, index_cos, roc_value
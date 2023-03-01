#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import math
import scipy.signal


# defined gabor filter
def gabor(x, y, delx, dely, f):
    '''
    x, y: index value from kernel
    delx: delta x
    dely: delta y
    f: frequency
    '''
    
    # gabor function
    g = 1 / (2 * np.pi * delx * dely) * np.exp(-0.5 * (x ** 2 / delx **2 + y ** 2 / dely ** 2)) * np.cos(
        2 * np.pi * f * math.sqrt(x ** 2 + y ** 2))
    return g


# 9*9 filter block
def filter_block(delx, dely, f):
    '''
    delx: delta x
    dely: delta y
    f: frequency
    '''
    temp_filter = np.zeros((9, 9))
    # apply gabor function on every block
    for i in range(9):
        for j in range(9):
            temp_filter[i, j] = gabor((-4 + i), (-4 + j), delx, dely, f)
    return temp_filter


def feature_extraction(enhanced):
    '''
    enhanced: enhanced version of normalized iris
    '''
    # ROI
    enhanced = enhanced[0:48, :]

    # chosen filters
    filter1 = filter_block(3, 1.5, 1 / 1.5)
    filter2 = filter_block(4.5, 1.5, 1 / 1.5)

    # convolution on filters defined above
    filtered1 = scipy.signal.convolve2d(enhanced,filter1,mode='same')
    filtered2 = scipy.signal.convolve2d(enhanced,filter2,mode='same')

    # Get the feature for each 8*8 block
    feature = []
    a = 1
    b = 1
    while a < 7:
        b = 1
        while b < 65:
            feature1 = abs(filtered1[(a - 1) * 8:a * 8, (b - 1) * 8:b * 8])
            feature2 = abs(filtered2[(a - 1) * 8:a * 8, (b - 1) * 8:b * 8])
            mean1 = feature1.mean()
            mean2 = feature2.mean()
            std1 = abs(feature1 - mean1).mean()
            std2 = abs(feature2 - mean2).mean()
            feature.append(mean1)
            feature.append(std1)
            feature.append(mean2)
            feature.append(std2)
            b += 1
        a += 1
    return feature

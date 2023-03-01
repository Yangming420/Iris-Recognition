#!/usr/bin/env python
# coding: utf-8

import os
import cv2
import numpy as np


def iris_normalization(img, pupil, iris):
    '''
    img: gray image of iris
    pupil: pupil position parameters with form (x-value, y-value, radius)
    iris: iris position paramaters with form (x-value, y-value, radius)
    '''
    
    # initialize normalized image with shape 64*512
    M, N = 64, 512
    norm_img = np.zeros(shape=(M, N))
    
    # find the corresponding point combining formula from the paper with the one from ppt
    for i in range(N):
        theta = 2 * np.pi * i / N
        for j in range(M):
            x = (1 - j / M) * (pupil[0] + np.cos(theta) * pupil[2]) + (
                            iris[0] + np.cos(theta) * iris[2]) * j / M
            y = (1 - j / M) * (pupil[1] + np.sin(theta) * pupil[2]) + (
                            iris[1] + np.sin(theta) * iris[2]) * j / M
            x = int(x)
            y = int(y)
            
            # ignore point out of range
            try:
                norm_img[j, i] = img[y, x]
            except IndexError:
                pass
            continue
    norm_img = np.flip(norm_img, axis=1)
    return norm_img

def rotation(normed, degree):
    '''
    normed: normalized image with shape (64, 512)
    degree: the degree that original iris rotates
    '''
    # get the rotated image
    pixel = int(degree/360*512)
    rotate = np.hstack((normed[:,pixel:],normed[:,:pixel]))
    return rotate

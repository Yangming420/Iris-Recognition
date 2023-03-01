#!/usr/bin/env python
# coding: utf-8

import os
import cv2
import numpy as np
import math


def iris_enhancement(normed):
    '''
    normed: normalized iris image with shape (64, 512)
    '''
    # ROI of value in [75, 155]
    mask = normed.copy()
    for i in range(len(normed)):
        for j in range(len(normed[0])):
            if normed[i, j] < 75 or normed[i, j] > 155:
                mask[i, j] = 0
            else:
                mask[i, j] = 1

    normed = normed * mask

    # 16*16 background illumination estimate
    background = np.zeros((4, 32))
    a = 1
    b = 1
    while a < 5:
        b = 1
        while b < 33:
            temp = normed[(a - 1) * 16:a * 16, (b - 1) * 16:b * 16]
            background[a - 1, b - 1] = temp.mean()
            b += 1
        a += 1

    # bicubic interpolation
    dim = (512, 64)
    bicubic = cv2.resize(np.array(background), dim, interpolation=cv2.INTER_CUBIC)

    # histogram equalization in each 32 * 32 region
    normed = normed - bicubic
    a = 1
    b = 1
    while a < 3:
        b = 1
        while b < 17:
            temp = normed[(a - 1) * 32:a * 32, (b - 1) * 32:b * 32]
            normed[(a - 1) * 32:a * 32, (b - 1) * 32:b * 32] = cv2.equalizeHist(temp.astype(np.uint8))
            b += 1
        a += 1
    return normed
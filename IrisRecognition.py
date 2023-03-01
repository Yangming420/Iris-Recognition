#!/usr/bin/env python
# coding: utf-8

import cv2
import glob
from IrisLocalization import *
from IrisNormalization import *
from ImageEnhancement import *
from FeatureExtraction import *
from IrisMatching import *
from PerformanceEvaluation import *


# extract images from files
def img_extraction(imgpath):
    '''
    imgpath: image path
    '''
    image_list = []
    for filename in glob.glob(imgpath):
        im = cv2.imread(filename)
        image_list.append(im)
    return image_list


# read the image into two lsits
train_imgpath = './CASIA Iris Image Database (version 1.0)/*/1/*.bmp'
train = img_extraction(train_imgpath)
test_imgpath = './CASIA Iris Image Database (version 1.0)/*/2/*.bmp'
test = img_extraction(test_imgpath)

# convert color image to gray image
for i in range(len(train)):
    train[i] = cv2.cvtColor(train[i], cv2.COLOR_BGR2GRAY)
for i in range(len(test)):
    test[i] = cv2.cvtColor(test[i], cv2.COLOR_BGR2GRAY)

# preprocess image using defined functions
# iris enhancement has a negative impact on our predictions, so we do not use it
norm_train = []
x_train = []
degree = [-3, -2, -1, 0, 1, 2, 3]
# expand one image to seven images by rotating 7 different degrees
for i in range(len(train)):
    img = train[i]
    pupil, iris = iris_localization(img)
    normed = iris_normalization(img, pupil, iris)
    norm_train.append(normed)
# get features
for i in range(len(norm_train)):
    for j in range(len(degree)):
        rotate = rotation(norm_train[i], degree[j])
        # normed = iris_enhancement(rotate)
        feature = feature_extraction(rotate)
        x_train.append(feature)

x_test = []
for i in range(len(test)):
    img = test[i]
    pupil, iris = iris_localization(img)
    normed = iris_normalization(img, pupil, iris)
    # normed = iris_enhancement(normed)
    feature = feature_extraction(normed)
    x_test.append(feature)

# manually compute labels for images
y_train = []
for i in range(1, 109):
    for j in range(0, 21):
        y_train.append(i)

y_test = []
for i in range(1, 109):
    for j in range(0, 4):
        y_test.append(i)
y_test = np.array(y_test)

# Compute CRR using original features
n = len(test)
od1, od2, od3, roc = iris_match(x_train, x_test)
oCRR1, oCRR2, oCRR3 = CRR(od1, od2, od3, y_test, n)
OCRR = [oCRR1, oCRR2, oCRR3]

# Compute CRR for different dimensions
reduced_dim = [30, 40, 50, 60, 70, 80, 90, 107]
CRR_d3 = []
for i in range(len(reduced_dim)):
    redx_train, redx_test = lda_dim_reduction(x_train, y_train, x_test, reduced_dim[i])
    d1, d2, d3, roc = iris_match(redx_train, redx_test)
    CRR1, CRR2, CRR3 = CRR(d1, d2, d3, y_test, n)
    CRR_d3.append(CRR3 * 100)

# Compute CRR using 107 dimensions
redx_train, redx_test = lda_dim_reduction(x_train, y_train, x_test, 107)
d1, d2, d3, roc = iris_match(redx_train, redx_test)
CRR1, CRR2, CRR3 = CRR(d1, d2, d3, y_test, n)
DCRR = [CRR1, CRR2, CRR3]

# FMR, FNMR for ROC table
threshold = [0.446, 0.472, 0.502]
FMR, FNMR = [], []
for i in range(len(threshold)):
    fm, fnm = ROC(d3, y_test, roc, threshold[i])
    FMR.append(fm)
    FNMR.append(fnm)

# FMR, FNMR for ROC plot
threshold_p = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
FMR_p, FNMR_p = [], []
for i in range(len(threshold_p)):
    fm, fnm = ROC(d3, y_test, roc, threshold_p[i])
    FMR_p.append(fm)
    FNMR_p.append(fnm)

# CRR table
CRR_table(OCRR, DCRR)

# CRR plot
CRR_plot(reduced_dim, CRR_d3)

# ROC table
ROC_table(threshold, FMR, FNMR)

# ROC plot
ROC_plot(FMR_p, FNMR_p)

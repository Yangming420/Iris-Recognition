#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import math
import tabulate
import matplotlib.pyplot as plt

# calculate CRR
def CRR(d1, d2, d3, y_test, n):
    '''
    d1: matching index using l1 distance
    d2: matching index using l2 distance
    d3: matching index using cosine similarity measure
    y_test: labels of test image
    n: total number of test image
    '''
    # convert index to labels
    for i in range(len(d1)):
        d1[i] = d1[i]//21+1
        d2[i] = d2[i]//21+1
        d3[i] = d3[i]//21+1
    
    count1 = 0
    count2 = 0
    count3 = 0
    
    # calculate the number of correct match
    for i in range(len(d1)):
        if y_test[i] == d1[i]:
            count1+=1
        if y_test[i] == d2[i]:
            count2+=1
        if y_test[i] == d3[i]:
            count3+=1
            
    # calculate CRR
    CRR_1 = count1/n
    CRR_2 = count2/n
    CRR_3 = count3/n
    return CRR_1, CRR_2, CRR_3


def CRR_table(OCRR, CRR):
    '''
    OCRR: CRR calculated by using original features
    CRR: CRR calculated by using features after dimension reduction
    '''
    print("     Recognition Result Using Different Similarity Measures")
    print(tabulate.tabulate([['L1 distance measure', OCRR[0]*100, CRR[0]*100 ],
                   ['L2 distance measure', OCRR[1]*100, CRR[1]*100 ],
                   ['Cosine similarity measure', OCRR[2]*100, CRR[2]*100 ]],
                  headers=['Similartiy measure','Original feature set(%)', "Reduced feature set(%)"]))

def CRR_plot(reduced_dim, CRR_d3):
    '''
    reduced_dim: the set of numbers that we use for dimension reduction
    CRR_d3: CRR calculated by using cosine similarity measure
    '''
    plt.figure()
    plt.plot(reduced_dim,CRR_d3)
    plt.xlabel('Dimensionality of the feature vector')
    plt.ylabel('Correct recognition rate')
    plt.title('Recognition results using features of different dimensionality')
    plt.savefig('CRR_plot.png')
    plt.show()
    
def ROC(d3, y_test, roc_value, threshold):
    '''
    d3: matching result using cosine similarity measure
    y_test: test image labels
    roc_value: the minimum cosine similarity measure for each test image
    threshold: threshold for calculating FMR and FNMR
    '''
    FMR = 0
    FNMR = 0
    # Convert results to corresponding labels
    for i in range(len(d3)):
        d3[i] = d3[i]//21+1
        
    # calculate FMR, FNMR
    for i in range(len(d3)):
        if y_test[i] == d3[i] and roc_value[i] >= threshold:
            FNMR += 1
        elif y_test[i] != d3[i] and roc_value[i] <= threshold:
            FMR += 1
    return FMR/len(y_test), FNMR/len(y_test)

def ROC_table(threshold, FMR, FNMR):
    '''
    threshold: the set of threshold numbers
    FMR: set of FMR with corresponding threshold
    FNMR: set of FNMR with corresponding threshold
    '''
    print("False Match and False Nonmatch Rates with Different Threshold Values")
    print(tabulate.tabulate([[threshold[0], FMR[0],FNMR[0]], 
                    [threshold[1], FMR[1],FNMR[1]],
                    [threshold[2], FMR[2],FNMR[2]]],
                    headers=['Threshold', 'False match rate',"False non-match rate"]))
    
def ROC_plot(FMR, FNMR):
    '''
    FMR: set of FMR values
    FNMR: set of FNMR values
    '''
    plt.figure()
    plt.plot(FMR,FNMR)
    plt.xlabel('False Match Rate')
    plt.ylabel('False Non_match Rate')
    plt.title('ROC Curve')
    plt.savefig('ROC_plot.png')
    plt.show() 

# Iris Recognition

This project is to implement an iris recognition algorithm. The ideas are mainly from Li Ma's essay.

Explain the whole logic of design(The whole procedure)
------

1. Iris Localization

First, we try to find the position of pupil. 

(a)：convert the gray image to binary image

(b): use Canny edge detector and remove noise

(c): use Hough circle to locate pupil

Then, we try to find the position of iris.

(a): use a zoomed image to improve performance. Max size 200*240

(b): use median blur and canny edge detector to find rough contours

(c): manually remove noise and use Hough circle to location iris

2. Iris Normalization

(a): unwrap the iris ring to a rectangle with size 64*512 based on formula from Li Ma's essay

(b): define a rotation function that can find the normalized rectangle after rotation

3. Image Enhancement

(a): get the mean of each 16*16 block and do bicubic interpolation to estimate the background

(b): subtract the background from normalized image

(c): do histogram equalization on each 32*32 block

4. Feature Extraction

(a): use 48*512 ROI for better performance 

(b): create Gabor kernel function and define the 9*9 filter blocks

(c): perform 2d convolution by using filters defined and find the mean and sd of each 8*8 block as features. The total number of features for one image is 1536.

5. Iris Matching

(a): use the lda to reduce dimension on feature vectors 

(b): use the nearest center classifier to get the predictions


6. Performance Evaluation

(a): calculate CRR based on predictions we have from iris matching 

(b): use cosine similarity measure to and some thresholds to find FMR(false match rate) and FNMR(false non-match rate) 

(c): create CRR table, CRR plot, ROC table, ROC plot

7. Iris Recognition

This is the main part of the project. We use results from above sections to see our performance.

Limitation and improvement
-------

1. The CRR we found are all less than 90%, and there is a huge difference with the outcome from the paper.

2. The iris localization part can catch the iris ring but not perfectly. Generally our outter boundaries are larger and contain eyelids. Since all these would affect our project a lot, we may want to get rid of eyelids and tune the parameters to find a better iris ring.

3. Recently iris recognition can be done using CNN or some other deep learning algorithms. The efficiency will get a better improvement.

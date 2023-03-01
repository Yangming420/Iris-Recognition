import cv2
import numpy as np


def iris_localization(img_gray):
    '''
    img_gray: the gray image of iris
    '''
    
    # Find the location of pupil
    # convert to binary image
    _, img_binary = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY)

    # morphological operations
    kernel = np.ones((5, 5), np.uint8)
    img_dilate = cv2.dilate(img_binary, kernel)
    img_morph = cv2.erode(img_dilate, kernel)

    # Canny edge detector
    pupil_edge = cv2.Canny(img_morph, 10, 20)

    # find the contours
    contours, hierarchy = cv2.findContours(image=pupil_edge, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    new_contours = []
    # move contours with length < 150 to new_contours as noise
    for i in range(len(contours)):
        if len(contours[i]) < 150:
            new_contours.append(contours[i])

    # assign the surrounding parts and noise defind above to be 0
    for i in range(len(new_contours)):
        for j in range(len(new_contours[i])):
            pupil_edge[new_contours[i][j, 0][1], new_contours[i][j, 0][0]] = 0

    for i in range(0, 280):
        for j in range(280, 320):
            pupil_edge[i, j] = 0
    for i in range(0, 280):
        for j in range(0, 40):
            pupil_edge[i, j] = 0

    # Use Hough Circle to find the position of pupil
    circle = cv2.HoughCircles(pupil_edge, cv2.HOUGH_GRADIENT, 1, 200,
                              param1=20, param2=0.9, minRadius=30, maxRadius=60)
    circle = np.uint16(np.around(circle))
    pupil = [i for i in circle[0, 0]]

    # Find the location of iris
    # Use max 200*240 zoomed image for better performance
    row = 100
    col = 120
    if pupil[1] < row:
        row = pupil[1]
    elif 280 - pupil[1] < row:
        row = 280 - pupil[1]
    if pupil[0] < col:
        col = pupil[0]
    elif 320 - pupil[0] < col:
        col = 320 - pupil[0]

    img_zoom = img_gray[pupil[1] - row:pupil[1] + row, pupil[0] - col:pupil[0] + col]

    # Use median blur to reduce noise
    img_blur = cv2.medianBlur(img_zoom, 9)

    # Canny edge detector
    iris_edge = cv2.Canny(img_blur, 20, 30)

    # manually remove regions above, bottom and around pupil
    for i in range(len(iris_edge)):
        for j in range(len(iris_edge[0])):
            if i <= row - 20:
                iris_edge[i, j] = 0
            if i <= row + pupil[2] + 35 and col - pupil[2] - 35 <= j <= col + pupil[2] + 35:
                iris_edge[i, j] = 0
            elif i >= row + pupil[2] + 20:
                iris_edge[i, j] = 0

    # Use Hough Circle to find the position of iris
    circle = cv2.HoughCircles(iris_edge, cv2.HOUGH_GRADIENT, 1, 200,
                              param1=30, param2=0.9, minRadius=90, maxRadius=118)
    circle = np.uint16(np.around(circle))
    iris = [i for i in circle[0, 0]]

    # Map the iris parameter back to original image
    iris[0] = pupil[0] - col + iris[0]
    iris[1] = pupil[1] - row + iris[1]

    return pupil, iris

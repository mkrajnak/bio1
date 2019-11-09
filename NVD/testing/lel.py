#!/usr/bin/env python3
from os import listdir

import cv2
import scipy.ndimage as ndimage
import numpy as np

if __name__ == "__main__":
    for i in [x for x in listdir('.') if ('jpg' in x or 'png' in x)]:
        # reading grayscale image
        orig = cv2.imread(i)
        # grayscale
        gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        # blur
        gray_blur = cv2.GaussianBlur(gray, (5,5), 0)
        cv2.imwrite(f'results/{i}_2.jpg', gray_blur)#,edge)
        # filtering the disc, forked from entrega.py
        # _, thresh = cv2.threshold(gray_blur, 127, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        # gray = cv2.addWeighted(gray, 1.5, gray_blur, -0.5, 0, gray)
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(31,31))
        # gray = ndimage.grey_closing(gray,structure=kernel)
        # gray = cv2.equalizeHist(gray)
        print(f'mean: {np.median(gray_blur)}')
        _, thresh = cv2.threshold(gray, 120, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        # inverting the image to get better contours
        thresh = cv2.bitwise_not(thresh)
        cv2.imwrite(f'results/{i}_3.jpg', thresh)

        contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            print(f'Image: {i} not suitable for automatic detection, cannot detect disc for img')
        
        contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
        cnt = max(contour_sizes, key=lambda x: x[0])[1]

        area = cv2.contourArea(cnt)
        if area > 100000:
            print(f'Image: {i} not suitable for automatic detection, area is too light for proper disc detection')

        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [cnt], 0, 255, -1)
        cv2.imwrite(f'results/{i}_4.jpg', cv2.drawContours(orig, [cnt],-1,(0,255,0),3))

        approx = cv2.approxPolyDP(cnt,0.1*cv2.arcLength(cnt,True),True)
        cv2.drawContours(orig, [approx], -1, (0, 0, 255), 3)
        

        (x,y),radius = cv2.minEnclosingCircle(cnt)
        center = (int(x),int(y))
        radius = int(radius)
        cv2.circle(orig,center,radius,(0,255,0),2)
        cv2.imwrite(f'results/{i}_5.jpg', orig)

        # # # edge detection
        # edge = cv2.Canny(thresh, 100, 200)
        # blured = cv2.medianBlur(edge, 15)

        # detecting the brightest pixel 
        # _, _, _, maxLoc = cv2.minMaxLoc(gray)
        # cv2.circle(gray, maxLoc, 50, (0,255,0), 3)

        # detect circles in the image
        # circles = cv2.HoughCircles(edge, cv2.HOUGH_GRADIENT, 1.5, 200, minRadius=100, maxRadius=350)

        # # ensure at least some circles were found
        # if circles is not None:
        #     # convert the (x, y) coordinates and radius of the circles to integers
        #     circles = np.round(circles[0, :]).astype("int")

        #     # loop over the (x, y) coordinates and radius of the circles
        #     for (x, y, r) in circles:
        #         # draw the circle in the output image, then draw a rectangle
        #         # corresponding to the center of the circle
        #         cv2.circle(output, (x, y), r, (0, 255, 0), 4)
        #         cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        
        # cv2.imwrite(f'results/{i}_bright.jpg', np.hstack([gray, thresh]))#,edge)
        # cv2.imwrite(f'results/{i}_bright.jpg', cv2.drawContours(gray,contours,-1,(0,255,0),3))#,edge)
        print(f'img {i} done')
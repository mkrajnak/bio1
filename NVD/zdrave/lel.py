#!/usr/bin/env python3
from os import listdir

import cv2
import scipy.ndimage as ndimage
import numpy as np

# multiply the disc area to cover area around the disc
RADIUS_MULTIPLIER = 1.3


''' Simple error wrapped to for unification of error strings'''
def err(img, str):
    print(f'Image: {img} not suitable for automatic detection:{str}')


'''
    return the list of all light areas located in the image, of of them
    is the area of disc
'''
def get_contours(img):
    # grayscale + blur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blured = cv2.GaussianBlur(gray, (7,7), 0)
    # calculate threshold (white on black)
    _, thresh = cv2.threshold(blured, 120, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    # inverting the image to get better contours (black on white)
    thresh = cv2.bitwise_not(thresh)
    # calculating detected areas
    contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return contours


'''calculate the area of all contours and return the biggest one (disc)'''
def get_biggest_contour_area(contours):
    area, contour = max(
        [(cv2.contourArea(x), x) for x in contours], key=lambda x: x[0])
    return (area, contour)


'''
    calculate the minimal circle around contour, in most of the cases
    we get a circle around the whole disc, returned radius is mupltiplied
    to make sure the surrounding area around disc is covered as well
'''
def get_min_circle(cnt):
    (x,y), radius = cv2.minEnclosingCircle(cnt)
    # fix darker images, where radius is too small
    # print(f'radius:{radius}')
    if radius < 99:
        # area too small, let's cover more
        return (int(x),int(y)), int(radius * 3)
    elif radius < 140:
        # smaller area, different multiplier
        return (int(x),int(y)), int(radius * 2)
    else:
        return (int(x),int(y)), int(radius * RADIUS_MULTIPLIER)


if __name__ == "__main__":
    for i in [x for x in listdir('.') if ('jpg' in x or 'png' in x)]:
        orig = cv2.imread(i)
        contours = get_contours(orig)
        if len(contours) == 0:
            err(i, 'cannot detect disc for img')
            continue

        area, cnt =  get_biggest_contour_area(contours)
        # if area > 100000:
        #     err(i, ' area is too light for proper disc detection')
        #     continue

        center, radius = get_min_circle(cnt)
        cv2.circle(orig,center,radius,(0,255,0),2)
        cv2.imwrite(f'results/{i}_radius.jpg', orig)


        print(f'img {i} done')

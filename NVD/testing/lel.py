#!/usr/bin/env python3
from os import listdir
from math import pi

import cv2
import scipy.ndimage as ndimage
import numpy as np

# multiply the disc area to cover area around the disc
RADIUS_MULTIPLIER = 1.2
MAX_DISC_RADIUS = 200

''' Simple error wrapped to for unification of error strings'''
def err(img, str):
    print(f'Image: {img} not suitable for automatic detection:{str}')


'''
    return the list of all light areas located in the image, of of them
    is the area of disc
'''
def get_contours(i):
    img = cv2.imread(i)
    # grayscale + blur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blured = cv2.GaussianBlur(gray, (7,7), cv2.BORDER_DEFAULT)
    # cv2.imwrite(f'results/{i}_bfr.jpg', blured)
    mean = int(np.mean(blured, axis=(0, 1)))
    print(f'mean {mean}')
    if mean > 74:
        gray = cv2.addWeighted(gray, 1, blured, -0.3, 0, gray)
        offset = 70
    else:
        offset = 60
        gray = cv2.addWeighted(gray, 1.2, blured, -0.5, 0, gray)

    mean = int(np.mean(gray, axis=(0, 1)))
    blured = cv2.GaussianBlur(gray, (7,7), cv2.BORDER_DEFAULT)

    # calculate threshold (white on black)
    _, thresh = cv2.threshold(blured, mean+offset, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    # inverting the image to get better contours (black on white)
    thresh = cv2.bitwise_not(thresh)
    # cv2.imwrite(f'results/{i}_thrsh.jpg', thresh)
    # calculating detected areas
    contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return img, contours


'''calculate the area of all contours and return the biggest one (disc)'''
def get_biggest_contour_area(contours):
    area, contour = max(
        [(cv2.contourArea(x), x) for x in contours], key=lambda x: x[0])
    return contour


'''
    calculate the minimal circle around contour, in most of the cases
    we get a circle around the whole disc, returned radius is mupltiplied
    to make sure the surrounding area around disc is covered as well
'''
def get_min_circle(cnt):
    (x,y), radius = cv2.minEnclosingCircle(cnt)
    # fix darker images, where radius is too small
    print(f'radius:{radius}')
    if radius > MAX_DISC_RADIUS:
        return (None, radius)
    elif radius < 30:
        return (int(x),int(y)), int(radius * 10)
    elif radius < 99:
        # area too small, let's cover more
        return (int(x),int(y)), int(radius * 3)
    elif radius < 140:
        # smaller area, different multiplier
        return (int(x),int(y)), int(radius * 2)
    else:
        return (int(x),int(y)), int(radius * RADIUS_MULTIPLIER)


def get_vessels(image):
	b,green_fundus,r = cv2.split(image)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	contrast_enhanced_green_fundus = clahe.apply(green_fundus)

	# applying alternate sequential filtering (3 times closing opening)
	r1 = cv2.morphologyEx(contrast_enhanced_green_fundus, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
	R1 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
	r2 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
	R2 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
	r3 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)
	R3 = cv2.morphologyEx(r3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)
	f4 = cv2.subtract(R3,contrast_enhanced_green_fundus)
	f5 = clahe.apply(f4)
	# cv2.imwrite(destinationFolder+file_name_no_extension+"_f5.png", f5)

	# removing very small contours through area parameter noise removal
	ret,f6 = cv2.threshold(f5,15,255,cv2.THRESH_BINARY)
	mask = np.ones(f5.shape[:2], dtype="uint8") * 255
	contours, hierarchy = cv2.findContours(f6.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	for cnt in contours:
		if cv2.contourArea(cnt) <= 200:
			cv2.drawContours(mask, [cnt], -1, 0, -1)
	im = cv2.bitwise_and(f5, f5, mask=mask)
	ret,fin = cv2.threshold(im,15,255,cv2.THRESH_BINARY_INV)
	newfin = cv2.erode(fin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)
	# removing blobs of unwanted bigger chunks taking in consideration they are not straight lines like blood
	#vessels and also in an interval of area
	fundus_eroded = cv2.bitwise_not(newfin)
	# cv2.imwrite(destinationFolder+file_name_no_extension+"_newfin.png", f5)
	xmask = np.ones(image.shape[:2], dtype="uint8") * 255
	xcontours, xhierarchy = cv2.findContours(fundus_eroded.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	for cnt in xcontours:
		shape = "unidentified"
		peri = cv2.arcLength(cnt, True)
		approx = cv2.approxPolyDP(cnt, 0.04 * peri, False)
		if cv2.contourArea(cnt) <= 1500:
			 cv2.drawContours(xmask, [cnt], -1, 0, -1)

	return cv2.bitwise_and(fundus_eroded,fundus_eroded,mask=xmask)



if __name__ == "__main__":
    for i in [x for x in listdir('.') if ('jpg' in x or 'png' in x)]:
        img, contours = get_contours(i)
        if len(contours) == 0:
            err(i, 'cannot detect disc for img')
            continue

        cnt = get_biggest_contour_area(contours)

        center, radius = get_min_circle(cnt)
        if not center:
            err(i, f' radius {radius} too large')
            continue

        vessels = get_vessels(img)
        mask = np.zeros(vessels.shape, np.uint8)
        cv2.circle(mask,center,radius,(255,255,255),-1)
        vessels = cv2.bitwise_and(vessels,vessels,mask=mask)

        rectX = center[0] - radius
        rectY = center[1] - radius
        vessels = vessels[rectY:(rectY+2*radius), rectX:(rectX+2*radius)]
        cv2.imwrite(f'results/{i}_radius.jpg', vessels)

        nonzero = cv2.countNonZero(vessels)
        total = pi*(radius**2)
        zero = total - nonzero
        ratio = zero * 100 / total

        if ratio == 100:
            print('No veins on disc detected')
            continue

        if ratio < 80:
            print('Chance of NVD')

        print(f'img {i} done, ratio {ratio}')

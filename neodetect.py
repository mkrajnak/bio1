#!/usr/bin/env python3
from os import listdir
from os.path import isdir
from math import pi
from xml.etree.ElementTree import Element, SubElement, Comment, tostring
from xml.dom import minidom

import argparse
import cv2
import numpy as np
import sys

# multiply the disc area to cover area around the disc
RADIUS_MULTIPLIER = 1.2
MAX_DISC_RADIUS = 500
DEBUG = False


''' debug print wrapper, debug messages are passed only if --debug arg is passed'''
def debug_print(msg):
    if DEBUG:
        print(msg)


''' Simple error wrapped to for unification of error strings'''
def err(img, str):
    debug_print(f'Image: {img} not suitable for automatic detection:{str}')


def help():
    print("Usage:\n python3 neodetect.py <folder> [--debug]")
    print("<folder> - mandatory, folder with .jpg, RGB images of fundus")
    print("[--debug] - optional, debug images will be printed during image processing")
    exit(1)

'''
    return the list of all light areas located in the image, of of them
    is the area of disc
'''
def get_disc_contours(i):
    img = cv2.imread(i)
    # grayscale + blur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blured = cv2.GaussianBlur(gray, (7,7), cv2.BORDER_DEFAULT)
    mean = int(np.mean(blured, axis=(0, 1)))
    debug_print(f'Mean for grayscale img:{mean}')
    # Make image more dark, so the light areas can stand out
    if mean > 74:
        gray = cv2.addWeighted(gray, 1, blured, -0.3, 0, gray)
        offset = 70
    else:
        gray = cv2.addWeighted(gray, 1.2, blured, -0.5, 0, gray)
        offset = 60
    # recalculate values
    mean = int(np.mean(gray, axis=(0, 1)))
    blured = cv2.GaussianBlur(gray, (7,7), cv2.BORDER_DEFAULT)
    # calculate threshold (white on black)
    _, thresh = cv2.threshold(blured, mean+offset, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    # inverting the image to get better contours (black on white)
    thresh = cv2.bitwise_not(thresh)
    return img, get_contours(thresh)


''' calculate the area of all contours and return the biggest one (disc)'''
def get_biggest_contour(contours):
    _, contour = max(
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
    debug_print(f'minEnclosingCircle radius:{radius}')
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


''' kernel wrapper '''
def get_kernel(params):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, params)


''' applying filter to idetify so only vessels should be left '''
def filter_step(img, kernel_param):
    open = cv2.morphologyEx(
        img, cv2.MORPH_OPEN, get_kernel(kernel_param), iterations=1)
    return cv2.morphologyEx(
        open, cv2.MORPH_CLOSE, get_kernel(kernel_param), iterations=1)


''' getting contours from the treshold img, input only black/white image'''
def get_contours(img):
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


'''
    blood vessels segmentation algorithm, forked from
    https://github.com/getsanjeev/retina-features
'''
def get_vessels(image):
    _, green, _ = cv2.split(image)
    # histogram equalisation
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = contrast_enhanced_green = clahe.apply(green)
    # applying alternate sequential filtering (3 times closing opening)
    for kernel in [(5,5), (11, 11),(23, 23)]:
        enhanced = filter_step(enhanced, kernel)
    final = clahe.apply(cv2.subtract(enhanced, contrast_enhanced_green))
    # removing very small contours through area parameter noise removal
    _, f6 = cv2.threshold(final, 15, 255, cv2.THRESH_BINARY)
    # initializing both masks
    xmask = mask = np.ones(final.shape[:2], dtype="uint8") * 255
    # drawing contours on the mask
    for cnt in get_contours(f6):
    	if cv2.contourArea(cnt) <= 50:
    		cv2.drawContours(mask, [cnt], -1, 0, -1)
    # creating image with drawn masks
    im = cv2.bitwise_and(final, final, mask=mask)
    _, fin = cv2.threshold(im, 15, 255, cv2.THRESH_BINARY_INV)
    newfin = cv2.erode(fin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations=1)
    fundus_eroded = cv2.bitwise_not(newfin)
    # removing smaller area which are probably not vessels
    for cnt in get_contours(fundus_eroded.copy()):
    	if cv2.contourArea(cnt) <= 1500:
            cv2.drawContours(xmask, [cnt], -1, 0, -1)
    return cv2.bitwise_and(fundus_eroded, fundus_eroded, mask=xmask)


''' count % ratio of white vs black pixels from vessel map '''
def get_ratio(vessels):
    nonzero = cv2.countNonZero(vessels)
    total = pi*(radius**2)
    return (total - nonzero) * 100 / total


def add_xml_subElement(parent, name, text):
    el = SubElement(parent, name)
    el.text = text
    return el


if __name__ == "__main__":
    if len(sys.argv) == 1 or len(sys.argv) > 3:
        help()
    else:
        if len(sys.argv) == 3 and '--debug' in sys.argv[2]:
            DEBUG = True
        if not isdir(sys.argv[1]):
            print(f'{sys.argv[1]} not a directory')
            help()
        fldr = sys.argv[1]

    # init xml
    root = Element('neodetect')
    root.set('version', '1.0')
    root.append(Comment('Generated by neodetect.py, result is not 100%, visit your doctor!'))

    for i in [x for x in listdir(fldr) if 'jpg' in x]:
        img_path = f'{fldr}/{i}'
        debug_print(f'*** Starting detection for img {img_path}')
        child = SubElement(root, 'img')
        add_xml_subElement(child, 'name', f'{img_path}')

        img, contours = get_disc_contours(img_path)
        if len(contours) == 0:
            err(i, 'cannot detect disc for img, skipping\n')
            add_xml_subElement(child, 'error', 'cannot detect disc for img, skipping')
            continue
        # the biggest contour should be disc area, obtaining ROI
        cnt = get_biggest_contour(contours)
        center, radius = get_min_circle(cnt)
        if not center:
            err(i, f' radius {radius} too large, skipping\n')
            add_xml_subElement(child, 'error', 'cannot detect disc for img, skipping')
            continue
        # cv2.imwrite(f'results/{i}_radius.jpg', cv2.circle(img, center, radius, (0,255,0),5))
        # getting vessel map, creating mask with ROI
        vessels = get_vessels(img)
        mask = np.zeros(vessels.shape, np.uint8)
        cv2.circle(mask, center, radius,(255,255,255),-1)
        vessels = cv2.bitwise_and(vessels, vessels, mask=mask)
        
        # cropping ROI from vessel map
        rectX = center[0] - radius
        rectY = center[1] - radius
        vessels = vessels[rectY:(rectY+2*radius), rectX:(rectX+2*radius)]
        # get % of pixel ratio difference
        ratio = get_ratio(vessels)
        # cv2.imwrite(f'results/{i}_vessels.jpg', vessels)

        if ratio == 100:
            debug_print('No veins on disc detected, skipping')
            add_xml_subElement(child, 'error', 'No vessels in area, skipping')
            continue
        
        if ratio < 78:
            debug_print('Chance of NVD')
            result = add_xml_subElement(child, 'result', '')
            add_xml_subElement(result, 'neovascularization', 'Neovascularization on the disc discdetected')
            add_xml_subElement(result, 'x', f'{center[0]}')
            add_xml_subElement(result, 'y', f'{center[1]}')
        else:
            add_xml_subElement(child, 'result', 'Healthy')
        
        debug_print(f'*** Img {i} done, ratio {ratio}\n')
    
    results = minidom.parseString(tostring(root)).toprettyxml(indent="   ")
    with open("results.xml", "w") as f:
        f.write(results)
    
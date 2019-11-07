import cv2
from os import listdir
import scipy.ndimage as ndimage


if __name__ == "__main__":
    for i in [x for x in listdir('.') if 'jpg' in x]:
        # reading grayscale image
        img = cv2.imread(i)
        _, img, _ = cv2.split(img)
        # applying gaussian blur to avoid noise
        img = cv2.GaussianBlur(img, (5, 5), 0)
        _, _, _, maxLoc = cv2.minMaxLoc(img)
        cv2.circle(img, maxLoc, 50, (0,255,0), 3)
        cv2.imwrite(f'results/{i}_bright.jpg', img)



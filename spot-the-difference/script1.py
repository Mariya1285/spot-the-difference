# import the necessary packages
from skimage import metrics
import argparse
import imutils
import cv2
import skimage

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

import warnings
warnings.filterwarnings("ignore", module="matplotlib")

import random 
 
def colors(n): 
  ret = [] 
  r = int(random.random() * 256) 
  g = int(random.random() * 256) 
  b = int(random.random() * 256) 
  step = 256 / n 
  for i in range(n): 
    r += step 
    g += step 
    b += step 
    r = int(r) % 256 
    g = int(g) % 256 
    b = int(b) % 256 
    ret.append((r,g,b))  
  return ret 


# load the two input images
image_orig = cv2.imread("cc01.jpg")

image_mod = cv2.imread("cc02.jpg")

resized_orig = image_orig    
resized_mod = image_mod


plt.imshow(resized_orig)


plt.imshow(resized_mod)

# convert the images to grayscale
gray_orig = cv2.cvtColor(resized_orig, cv2.COLOR_BGR2GRAY)
gray_mod = cv2.cvtColor(resized_mod, cv2.COLOR_BGR2GRAY)

# compute the Structural Similarity Index (SSIM) between the two
# images, ensuring that the difference image is returned
(score, diff) = skimage.metrics.structural_similarity(gray_orig, gray_mod, full=True)

diff = (diff * 255).astype("uint8")

print("Structural Similarity Index: {}".format(score))


# threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
thresh = cv2.threshold(diff, 0, 25,
    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)

cnts = imutils.grab_contours(cnts)

rgb= colors(len(cnts))
# loop over the contours
for index,c in enumerate(cnts):
# compute the bounding box of the contour and then draw the
# bounding box on both input images to represent where the two
# images differ
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(resized_orig, (x, y), (x + w, y + h), rgb[index], 2)
    cv2.rectangle(resized_mod, (x, y), (x + w, y + h), rgb[index], 2)
 

with warnings.catch_warnings():
  # show the output images

 #    cv2.imshow("window",resized_orig)
  # cv2.waitKey(0)
  cv2.imwrite("original-CC1.jpg",resized_orig)
  cv2.imwrite("modified-CC2.jpg",resized_mod)
  plt.imshow("Original", resized_orig)
  plt.imshow("Modified", resized_mod)
  cv2.imshow("Diff", diff)
  cv2.imshow("Thresh", thresh)

  cv2.waitKey(0)
#-*- coding:utf-8-*-
import matplotlib.pyplot as plt
import cv2
from skimage import io
import numpy as np

# Load image file
fpath = 'd:/github/ComputerVision/hw6/'
image = io.imread(fpath + 'checkerboard.JPG')
image_original = image.copy()

edges = cv2.Canny(image,50,200,apertureSize = 3)
gray = cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)

minLineLength = 100
maxLineGap = 0
# Perform the probabilistic Hough transform
lines = cv2.HoughLinesP(edges, 1, np.pi/360, 100, minLineLength, maxLineGap)

for i in range(len(lines)):
    for x1,y1,x2,y2 in lines[i]:
        cv2.line(image,(x1,y1),(x2,y2),(0,0,255),3)

plt.figure(figsize=(10, 5), dpi=150)
plt.subplot(1, 2, 1)
plt.imshow(image_original, cmap='gray', vmin=0, vmax=255)

plt.subplot(1, 2, 2)
plt.imshow(image, cmap='gray', vmin=0, vmax=255)

plt.show()

#-*- coding:utf-8-*-
import matplotlib.pyplot as plt
import cv2
from skimage import io
import numpy as np

# Load image file
fpath = 'd:/github/ComputerVision/hw6/'
image = io.imread(fpath + 'checkerboard.JPG')
image_original = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(image,50,150,apertureSize=3)

threshold = 100
# Perform the Hough transform
lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=threshold)

for i in range(len(lines)):
    for rho, theta in lines[i]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0+1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 -1000*(a))

        cv2.line(image,(x1,y1),(x2,y2),(0,0,255),2)

plt.figure(figsize=(10, 5), dpi=150)
plt.subplot(1, 2, 1)
plt.imshow(image_original, cmap='gray', vmin=0, vmax=255)

plt.subplot(1, 2, 2)
plt.imshow(image, cmap='gray', vmin=0, vmax=255)

plt.show()

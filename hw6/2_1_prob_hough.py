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
maxLineGap = 10
# Perform the probabilistic Hough transform
#lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength, maxLineGap)
lines = cv2.HoughLinesP(edges, 1, np.pi/360, 100, minLineLength, maxLineGap)

for i in range(len(lines)):
    for x1,y1,x2,y2 in lines[i]:
        cv2.line(image,(x1,y1),(x2,y2),(0,0,255),3)
        
maxLineGap_b = 20
lines_b = cv2.HoughLinesP(edges, 1, np.pi/360, 100, minLineLength, maxLineGap_b)
for i in range(len(lines_b)):
    for x1,y1,x2,y2 in lines_b[i]:
        cv2.line(image_original,(x1,y1),(x2,y2),(0,0,255),3)

plt.figure(figsize=(10, 5), dpi=150)
ax1 = plt.subplot(1, 2, 1)
ax1.set_title("maxLineGap: 10")
plt.imshow(image, cmap='gray', vmin=0, vmax=255)

ax2 = plt.subplot(1, 2, 2)
ax2.set_title("maxLineGap: 20")
plt.imshow(image_original, cmap='gray', vmin=0, vmax=255)
plt.show()

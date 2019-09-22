import matplotlib.pyplot as plt
from skimage import io, transform
import scipy
from scipy import ndimage
import numpy as np
import math


fpath = 'd:/lecture/computervision/'
image = io.imread(fpath + 'cameraman.tif')
imageTranslated = np.zeros(np.shape(image))

# make transform Translation -> Rotation -> Translation
iMax, jMax = np.shape(image)

Tx = - (0 + (iMax-1)) / 2
Ty = - (0 + (jMax-1)) / 2
Translation = np.array([[1,  0,  0],# Transform matrix for translation
                        [0,  1,  0],
                        [Tx, Ty, 1]])

Translation1 = np.array([[1,  0,  0],# Transform matrix for translation
                        [0,  1,  0],
                        [-Tx, -Ty, 1]])

Theta = math.radians(30)
Rotation = np.array([[np.cos(Theta), -np.sin(Theta), 0], # Transform matrix for rotation
                     [np.sin(Theta),  np.cos(Theta), 0],
                     [0,                          0, 1]])

#T = Translation # Transformation matrix
T = np.linalg.inv(Translation)
T_in = np.linalg.inv(Translation1)
T1 = np.linalg.inv(Rotation)

# apply transform
iTranslated = 0
while iTranslated < iMax:

    jTranslated = 0
    while jTranslated < jMax:
        
        translated = T_in.T.dot((iTranslated, jTranslated, 1)).astype(np.int)     
        translated = T1.dot((translated[0], translated[1], 1)).astype(np.int)
        translated = T.T.dot((translated[0], translated[1], 1)).astype(np.int)
        if 0 <= translated[0] < iMax and 0 <= translated[1] < jMax:
            imageTranslated[iTranslated, jTranslated] = image[translated[0], translated[1]]
        
        jTranslated = jTranslated + 1
    iTranslated = iTranslated + 1

# Check result
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')

plt.subplot(1, 2, 2)
plt.imshow(imageTranslated, cmap='gray')
plt.show()

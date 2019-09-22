import matplotlib.pyplot as plt
from skimage import io, transform
from scipy import ndimage
import numpy as np


fpath = 'd:/lecture/computervision/'
image = io.imread(fpath + 'cameraman.tif')
imageTranslated = np.zeros(np.shape(image))

# make transform x->x+15, y->y+30
Tx = 15
Ty = 30

T = np.array([[1., 0., 0.],
              [0., 1., 0.],
              [Tx, Ty, 1.]]) # Transformation matrix
T = np.linalg.inv(T)
#

# apply transform
iMax, jMax = np.shape(image)
iTranslated = 0
while iTranslated < iMax:

    jTranslated = 0
    while jTranslated < jMax:
        
        translated = T.T.dot((iTranslated, jTranslated, 1)).astype(np.int32)
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

""" Disparity map """
import cv2
import matplotlib.pyplot as plt


""" Load dataset """
fpath = 'd:/github/ComputerVision/hw4/'

img_L = cv2.imread(fpath + 'im0.png', cv2.IMREAD_GRAYSCALE)
img_R = cv2.imread(fpath + 'im1.png', cv2.IMREAD_GRAYSCALE)

""" Create disparity map """
nd = (16, 128)	# number of Disparities
bs = (5, 21) # block size
d_list = []

plt.figure(figsize=(10, 5), dpi=150)
i = 1
for n in nd:
    for b in bs:
        stereo = cv2.StereoBM_create(numDisparities=n, blockSize=b)
        disparity = stereo.compute(img_L, img_R)
        ax = plt.subplot(1, 4, i)
        plt.imshow(disparity)
        ax.title.set_text('nd=%d, bs=%d' %(n, b))
        i += 1

plt.show()

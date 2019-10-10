import matplotlib.pyplot as plt
from skimage import io, filters, util, morphology, measure
import numpy
from scipy import signal

def gkern(kernlen, nsig):
    # Return 2D Gaussian Kernel
    gkern1d = signal.gaussian(kernlen, std=nsig).reshape(kernlen, 1)
    kernel = numpy.outer(gkern1d, gkern1d)

    return kernel/kernel.sum()


def imageConvolution(image,kernel, median=False):

    ConvImage = numpy.zeros(numpy.shape(image))
    KernelSizeI, KernelSizeJ = numpy.shape(kernel)

    pad = (KernelSizeI - 1) // 2
    image = numpy.pad(image, pad, mode='constant', constant_values=0)
    for w in range(ConvImage.shape[0]):
        for h in range(ConvImage.shape[1]):
            if median:
                ConvImage[w,h] = numpy.median(image[(w):(w+KernelSizeI), (h):(h+KernelSizeI)] * kernel)
            else:
                ConvImage[w,h] = numpy.mean(image[(w):(w+KernelSizeI),(h):(h+KernelSizeI)] * kernel)
    return ConvImage.astype('uint8')

#################################################
################### Quiz 1 ######################
#################################################
PIXEL_MAX = 255
# Load image file
#fpath = '/store1/bypark/test_Comis/2_filter/'
fpath = 'd:/github/ComputerVision/hw2/'
image = io.imread(fpath + 'lena_gray.gif')


# make noise image
imageSaltAndPepper = util.noise.random_noise(image, mode='s&p')*PIXEL_MAX
imageSaltAndPepper = imageSaltAndPepper.astype('uint8')

# Kernel Definition
kernelSize = 3
sigma = 3
GaussianKernel = gkern(kernelSize, sigma)
MedianFilterWindow = morphology.square(kernelSize)

# Original Image
ax = plt.subplot(3, 2, 1)
ax.imshow(image, cmap='gray')
plt.title('Original Image')

# Salt & Pepper Noise Reduction
ax = plt.subplot(3, 2, 2)
ax.imshow(imageSaltAndPepper, cmap='gray')
plt.title('Gaussian Noise')

ax = plt.subplot(3, 2, 4)
filteredImage = imageConvolution(imageSaltAndPepper, GaussianKernel) # Gaussian filtering
ax.imshow(filteredImage, cmap='gray')
plt.title('Gaussian Filtering')

ax = plt.subplot(3, 2, 6)
filteredImage = imageConvolution(imageSaltAndPepper, MedianFilterWindow, median=True) # Median filtering
ax.imshow(filteredImage, cmap='gray')
plt.title('Median Filtering')

plt.show()

#################################################
################### Quiz 2 ######################
#################################################
for size in [5, 7]:
    kernelSize = size
    GaussianKernel = gkern(kernelSize, sigma)
    MedianFilterWindow = morphology.square(kernelSize)
    
    # Original Image
    ax = plt.subplot(3, 2, 1)
    ax.imshow(image, cmap='gray')
    plt.title('Original Image')
    
    # Salt & Pepper Noise Reduction
    ax = plt.subplot(3, 2, 2)
    ax.imshow(imageSaltAndPepper, cmap='gray')
    plt.title('Gaussian Noise')
    
    ax = plt.subplot(3, 2, 4)
    filteredImage = imageConvolution(imageSaltAndPepper, GaussianKernel) # Gaussian filtering
    ax.imshow(filteredImage, cmap='gray')
    plt.title('Gaussian Filtering')
    
    ax = plt.subplot(3, 2, 6)
    filteredImage = imageConvolution(imageSaltAndPepper, MedianFilterWindow, median=True) # Median filtering
    ax.imshow(filteredImage, cmap='gray')
    plt.title('Median Filtering')
    plt.show()

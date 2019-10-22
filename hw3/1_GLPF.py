import matplotlib.pyplot as plt
from skimage import io, filters, feature
import numpy
from scipy import fftpack, signal
import scipy


def gkern(kernlen, nsig):
    # Return 2D Gaussian Kernel
    gkern1d = signal.gaussian(kernlen, std=nsig).reshape(kernlen, 1)
    kernel = numpy.outer(gkern1d, gkern1d)

    return kernel

# Load image file
fpath = 'd:/github/ComputerVision/hw3/'
image = io.imread(fpath + 'character.tif').astype('float64')

# Fourier Transform
imageFFT = fftpack.fft2(image)
imageFFT = fftpack.fftshift(imageFFT)

# Gaussian Filter
sizeI, sizeJ = image.shape
d0 = sizeI/40  # cutoff fregquency, equivalent with sigma

GaussianFilter = gkern(sizeI, d0) # kernel

# Apply filter
imageFFTFiltered = imageFFT * GaussianFilter

# Inverse Fourier Transform
imageIFFT = fftpack.ifftshift(imageFFTFiltered)
imageFiltered = fftpack.ifft2(imageIFFT).astype(numpy.float32)

# Plot
plt.figure(figsize=(10, 10), dpi=150)
plt.subplot(3, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('OriginalImage', {'fontsize': 10})
plt.subplot(3, 3, 4)
plt.imshow(scipy.log10(1+scipy.absolute(imageFFT)), cmap='gray')
plt.title('Fourier', {'fontsize': 10})
plt.subplot(3, 3, 5)
plt.imshow(GaussianFilter, cmap='gray')
plt.title('GaussianFilter (D0 = ' + str(d0) + ')', {'fontsize': 10})
plt.subplot(3, 3, 6)
plt.imshow(scipy.log10(1+scipy.absolute(imageFFTFiltered)), cmap='gray')
plt.title('FilteredFourier', {'fontsize': 10})
plt.subplot(3, 3, 9)
plt.imshow(imageFiltered, cmap='gray')
plt.title('FilteredImage', {'fontsize': 10})

#plt.show()


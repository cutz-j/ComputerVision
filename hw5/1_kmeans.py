import matplotlib.pyplot as plt
from skimage import io
import numpy

# Load image file
fpath ='d:/github/ComputerVision/hw5/'
image = io.imread(fpath + 'wallpaper.jpg')

# image -> RGB vector conversion
sizeI, sizeJ, sizeK = image.shape
imageRGBvect = numpy.zeros((sizeI*sizeJ, sizeK))
imageRGBvectIdx = 0
i = 0
while i < sizeI:
    j = 0
    while j < sizeJ:
        imageRGBvect[imageRGBvectIdx, :] = image[i, j, :]
        imageRGBvectIdx += 1
        j += 1
    i += 1

# Kmeans initialization
k = 3
ClusterCenters = numpy.random.randint(low=0, high=255, size=(k, 3))

# Kmeans
convergeLimit = 10
deltaMeans = 99999
maxIter = 10000
iterN = 0
while deltaMeans > convergeLimit and iterN < maxIter:
    iterN += 1

    # Calculate every distance from center to points
    dist2Centers = numpy.sum([numpy.square(imageRGBvect - i) for i in ClusterCenters], axis=2).T

    # Cluster assignment to the closest center
    clustersAssigned = numpy.argmin(dist2Centers, axis=1)

    # ClusterCenter update
    NewClusterCenters = numpy.zeros((3,3))
    i = 0
    while i < k:
        clusterList = numpy.where(clustersAssigned == i)
        valueInlist = imageRGBvect[clusterList, ]
        NewClusterCenters[i, ] = valueInlist.mean(axis=1)
        i += 1

    # Calculate convergence
    deltaMeans = NewClusterCenters - ClusterCenters
    deltaMeans = numpy.square(deltaMeans)
    deltaMeans = numpy.sqrt(deltaMeans.sum(axis=1)).sum()
    print(deltaMeans)
    ClusterCenters = NewClusterCenters

# Convert image value to ClusterCenters
ClusterCenters = numpy.round(ClusterCenters).astype('uint8')

imageClustered = numpy.zeros(image.shape).astype('uint8')

imageRGBvectIdx = 0
i = 0
while i < sizeI:
    j = 0
    while j < sizeJ:
        imageClustered[i, j, :] = ClusterCenters[clustersAssigned[imageRGBvectIdx],:]
        imageRGBvectIdx += 1
        j += 1
    i += 1


# Plot
plt.figure(figsize=(20, 10), dpi=150)
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray', vmin=0, vmax=255)

plt.subplot(1, 2, 2)
plt.imshow(imageClustered, cmap='gray', vmin=0, vmax=255)

plt.show()


import cv2
import numpy
from matplotlib import pyplot as plt

def histogram(image):
    histogram = [0] * 256
    for pixel_intensity in image:
        histogram[pixel_intensity] += 1
    return histogram

def cummulative_sum(histogram):
    cummulative_sum = [0] * 256
    cummulative_sum[0] = histogram[0]
    index = 1
    for value in histogram[1:]:
        cummulative_sum[index] = cummulative_sum[index - 1] + value
        index += 1
    return cummulative_sum

def normalize_cummulative_sum(cummulative_sum):
    normalized_cummulative_sum = [0] * 256
    
    numerator_array = [0] * 256
    index = 0
    for entry in cummulative_sum:
        numerator_array[index] = (entry - cummulative_sum[0]) * 255
        index += 1

    denomenator = cummulative_sum[255] - cummulative_sum[0]

    index= 0
    for entry in normalized_cummulative_sum:
        normalized_cummulative_sum[index] = numerator_array[index] / denomenator
        index += 1

    return normalized_cummulative_sum

def set_new_intensity_values(normalized_cummulative_sum, image):
    image_new = [0] * len(image)
    index = 0
    for pixel_intensity in image:
        image_new[index] = normalized_cummulative_sum[pixel_intensity]
        index += 1
    return image_new

img_orig = cv2.imread('images.jpeg', 0)

img_orig_array = numpy.asarray(img_orig)
img_orig_array_flat = img_orig_array.flatten()
histogram_img_orig = histogram(img_orig_array_flat)

cummulative_sum = cummulative_sum(histogram_img_orig)
normalized_cummulative_sum = normalize_cummulative_sum(cummulative_sum)

image_new_array_flat = set_new_intensity_values(normalized_cummulative_sum, img_orig_array_flat)

image_new_array_flat = numpy.asarray(image_new_array_flat)
image_new_array = numpy.reshape(image_new_array_flat, img_orig_array.shape)

fig = plt.figure()

fig.add_subplot(221)
plt.title('Original Image')
plt.set_cmap('gray')
plt.imshow(img_orig_array)

fig.add_subplot(222)
plt.hist(img_orig.ravel(), 256, [0, 256])

fig.add_subplot(223)
plt.title('New Image')
plt.set_cmap('gray')
plt.imshow(image_new_array)

fig.add_subplot(224)
plt.hist(image_new_array.ravel(), 256, [0, 256])

plt.show(block=True)

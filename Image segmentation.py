# Image Segmentation using K-Means Clustering

# In computer vision, image segmentation is the process of partitioning an image into multiple segments. The goal of segmenting an image is to change the representation of an image into something that is more meaningful and easier to analyze. It is usually used for locating objects and creating boundaries.
# It is not a great idea to process an entire image because many parts in an image may not contain any useful information. Therefore, by segmenting the image, we can make use of only the important segments for processing.
# An image is basically a set of given pixels. In image segmentation, pixels which have similar attributes are grouped together. Image segmentation creates a pixel-wise mask for objects in an image which gives us a more comprehensive and granular understanding of the object.

# Uses:
# Used in self-driving cars. Autonomous driving is not possible without object detection which involves segmentation.
# Used in the healthcare industry. Helpful in segmenting cancer cells and tumours using which their severity can be gauged.

# Now, lets explore a method to read an image and cluster different regions of the image using the K-Means clustering algorithm and OpenCV.

# Color Clustering:
 
import numpy as np
import cv2
import matplotlib.pyplot as plt 

original_image = cv2.imread("C:/Users/shubhamv/Downloads/Image segmentation.jpeg")
print(original_image)
window_name = "Image"
cv2.imshow('image',original_image)

# We need to convert our image from RGB Colours Space to HSV to work ahead.
# According to wikipedia the R, G, and B components of an object’s color in a digital image are all correlated with the amount of light hitting the object, and therefore with each other, image descriptions in terms of those components make object discrimination difficult. Descriptions in terms of hue/lightness/chroma or hue/lightness/saturation are often more relevant.

pip install opencv-contrib-python

img = cv2.cvtColor(original_image, cv2.COLOR_RGB2HSV)

vectorized = img.reshape((-1,3))

# We convert the unit8 values to float as it is a requirement of the k-means method of OpenCV.
 
vectorized = np.float32(vectorized)    

# We are going to cluster with k = 3 because if you look at the image above it has 3 colors, green-colored grass and forest, blue sea and the greenish-blue seashore.

# Define criteria, number of clusters(K) and apply k-means()

criteria = (cv2.TERM_CRITERIA_EPS + c
            v2.TERM_CRITERIA_MAX_ITER,10,1)

# OpenCV provides cv2.kmeans(samples, nclusters(K), criteria, attempts, flags) function for color clustering.
# 1. samples: It should be of np.float32 data type, and each feature should be put in a single column.
# 2. nclusters(K): Number of clusters required at the end
# 3. criteria: It is the iteration termination criteria. When this criterion is satisfied, the algorithm iteration stops. Actually, it should be a tuple of 3 parameters. They are `( type, max_iter, epsilon )`:
# Type of termination criteria. It has 3 flags as below:
# cv.TERM_CRITERIA_EPS — stop the algorithm iteration if specified accuracy, epsilon, is reached.
# cv.TERM_CRITERIA_MAX_ITER — stop the algorithm after the specified number of iterations, max_iter.
# cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER — stop the iteration when any of the above condition is met.
# 4. attempts: Flag to specify the number of times the algorithm is executed using different initial labelings. The algorithm returns the labels that yield the best compactness. This compactness is returned as output.
# 5. flags: This flag is used to specify how initial centers are taken. Normally two flags are used for this: cv.KMEANS_PP_CENTERS and cv.KMEANS_RANDOM_CENTERS.

K = 3
attempts=10

ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)

# Now convert back into uint8.

center = np.uint8(center)

# Next, we have to access the labels to regenerate the clustered image

res = center[label.flatten()]

result_image = res.reshape((img.shape))

# Now let us visualize the output result with K=3

figure_size = 15
plt.figure(figsize=(figure_size,figure_size))
plt.subplot(1,2,1),plt.imshow(img)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(result_image)
plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])
plt.show()

# So the algorithm has categorized our original image into three dominant colors.

# Canny Edge detection: It is an image processing method used to detect edges in an image while suppressing noise.
# The Canny Edge detection algorithm is composed of 5 steps:
# Gradient calculation
# Non-maximum suppression
# Double threshold
# Edge Tracking by Hysteresis
# OpenCV provides cv2.Canny(image, threshold1,threshold2) function for edge detection.
# The first argument is our input image. Second and third arguments are our min and max threshold respectively.
# The function finds edges in the input image(8-bit input image) and marks them in the output map edges using the Canny algorithm. The smallest value between threshold1 and threshold2 is used for edge linking. The largest value is used to find initial segments of strong edges.


edges = cv2.Canny(img,150,200)
plt.figure(figsize=(figure_size,figure_size))
plt.subplot(1,2,1),plt.imshow(img)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()



































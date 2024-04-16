
import cv2
import numpy as np
import math
mask =  cv2.imread("stop_pose.png")
#mask[:240,:,:] = 0
edges = cv2.Canny(mask, 100, 200)
cv2.imwrite('stop_canny.jpg', edges)
#cv2.imshow('Binary Mask', edges)
#cv2.waitKey(0)
#cv2.destroyAllWindows

# Apply Hough Transform to detect lines
lines = cv2.HoughLines(edges, rho=4, theta=np.pi/180, threshold=100)
print(lines)
# Draw the lines on the original image
if lines is not None:
    for line in lines:
        rho, theta = line[0]
        degrees_value = math.degrees(theta)
        print(degrees_value)
        if 10 < np.abs(degrees_value) < 40 or (140) < np.abs(degrees_value) < (170) :
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(mask, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Display the result
cv2.imshow('Detected Lines', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
# Load the image
image = cv2.imread('your_image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Prewitt filter
prewittx = cv2.filter2D(image, cv2.CV_64F, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]))
prewitty = cv2.filter2D(image, cv2.CV_64F, np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]))

# Compute gradient magnitude
gradient_magnitude = np.sqrt(prewittx*2 + prewitty*2)

# Apply thresholding to create binary mask
threshold = 50  # Adjust threshold as needed
binary_mask = np.where(gradient_magnitude > threshold, 255, 0).astype(np.uint8)

# Display the binary mask
cv2.imshow('Binary Mask', binary_mask)
cv2.waitKey(0)
cv2.destroyAllWindows'''
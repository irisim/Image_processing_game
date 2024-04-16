
import cv2
import numpy as np
import math
#mask =  cv2.imread("stop_pose.png")
mask =  cv2.imread("no_slow.png")
#mask =  cv2.imread("stop_pose.png")
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

mask =  cv2.imread("no_slow.png")
#mask =  cv2.imread("stop_pose.png")

if True :#time.time() - Mario.time_up < 1 or time.time() - Mario.time_down < 1 :
    #mask = Mario.mask.copy ()
    mask[:100,:,:] = 0
    cv2.imshow('Detected Lines', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #if mask != None :
    #s = Mario.frame.shape()
    mask_lines = np.zeros_like(mask)
    added_line = np.zeros_like(mask)
    #Mario.mask_lines = mask_lines
    edges = cv2.Canny(mask, 100, 200)
    # Apply Hough Transform to detect lines
    lines = cv2.HoughLines(edges, rho=4, theta=np.pi/180, threshold=80)
    right_leg, left_leg = False, False
    right_x0_y0 = []
    left_x0_y0 = []
    # Draw the lines on the original image
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            degrees_value = math.degrees(theta)
            if 10 < np.abs(degrees_value) < 40 :

                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))

                right_leg = True
                cv2.line(added_line, (x1, y1), (x2, y2), (255, 0, 10), 2)
                mask_lines += added_line
                right_x0_y0.append((x0,y0))
                #print("right leg (x0,y0)=",(x0,y0))

            if (140) < np.abs(degrees_value) < (170) :
                left_leg = True
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(added_line, (x1, y1), (x2, y2), (255, 0, 10), 2)
                mask_lines += added_line
                left_x0_y0.append((x0, y0))
                #print("left leg (x0,y0)=", (x0, y0))
    '''if right_leg and left_leg :
        if ?? :
            Mario.stop = True
            Mario.mask_lines = mask_lines'''


    '''
    if right_leg and left_leg:
        intersection = np.where(mask_lines[:,:,3] > 1)
        if intersection[200:300,:] != 0 :
            for right_x0, _ in right_x0_y0:
                for left_x0, _ in left_x0_y0:
                    if right_x0 > left_x0:
                        Mario.stop = True
                        Mario.mask_lines = mask_lines
                        print("STOPPPPPPPPPPPPPPPPPP")
                        break'''
    #Mario.mask_lines = mask_lines
    cv2.imshow('Detected Lines', mask_lines)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if right_leg and left_leg:
        for right_x0, right_y0 in right_x0_y0:
            for left_x0, left_y0 in left_x0_y0:
                if right_x0 - left_x0 < 130 and np.abs(left_y0 - right_y0) > 60  :
                    intersection = np.where(mask_lines[:, :,2] > 11)  # Check for intersections in the blue channel (assuming lines are drawn in blue)
                    intersection_y_coords = intersection[0]
                    #print("intersection = ", intersection)
                    #print("intersection_y_coords = ", intersection_y_coords)
                    if len(intersection[0]) > 0:  # Check if any intersection points are found
                        intersection_y_coords = intersection[0]  # Y-coordinates of intersection points
                        #print("intersection_y_coords = ", intersection_y_coords)
                        if any(200 <= y <= 300 for y in intersection_y_coords):  # Check if any intersection point falls within the range 200 to 300

                            print("Intersection detected. STOPPPPPPPPPPPPPPPP")
                            print("right_x0, right_y0 =", right_x0, right_y0)
                            print("left_x0, left_y0",left_x0, left_y0)



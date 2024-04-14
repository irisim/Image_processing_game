import cv2
import numpy as np
from recorder import fake_cam
from webcam_stream import WebcamStream
import Frames_Process
import Frames_Process
from Player import Player
import sys
import os
SOURCE = 'fake cam'


def create_color_mask(frame, lower_bound, upper_bound, binary_mask):
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create mask for the specified color range
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    mask = cv2.bitwise_and(mask, binary_mask)
    return mask


def find_color_mass(mask, min_area_threshold=150):
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables to store center of mass and size
    center_of_mass = None
    size = 0

    # Loop over the contours
    for contour in contours:
        # Calculate the area of the contour
        area = cv2.contourArea(contour)

        # Check if the contour area exceeds the minimum threshold
        if area >= min_area_threshold:
            # Calculate the center of mass for each contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # Update center of mass and size if the current contour is larger
                if area > size:
                    center_of_mass = (cX, cY)
                    size = area

    return center_of_mass, size

def get_green(Mario):
    green_lower = np.array([40, 50, 50])
    green_upper = np.array([80, 255, 255])

    green_mask = create_color_mask(Mario.frame, green_lower, green_upper,Mario.mask_4color)
    green_center, green_size = find_color_mass(green_mask)
    return green_center, green_size

def get_red(Mario):
    # Define the color ranges (HSV format)
    red_lower1 = np.array([0, 180, 100])
    red_upper1 = np.array([10, 255, 255])

    red_lower2 = np.array([160, 100, 100])
    red_upper2 = np.array([179, 255, 255])

    red_mask1 = create_color_mask(Mario.frame, red_lower1, red_upper1,Mario.mask_4color)
    red_mask2 = create_color_mask(Mario.frame, red_lower2, red_upper2,Mario.mask_4color)
    # Combine masks for red color
    mask_red = cv2.bitwise_or(red_mask1, red_mask2)
    # Find center of mass and size for red and green colors separately
    red_center, red_size = find_color_mass(mask_red)
    return red_center, red_size

def get_green_and_red(Mario):
    Mario.frame_with_red_green = (Mario.frame).copy()
    green_center, green_size = get_green(Mario)
    Mario.green_center, Mario.green_size = green_center, green_size
    red_center, red_size = get_red(Mario)
    Mario.red_center, Mario.red_size = red_center, red_size
    # Draw circles at the center of masses
    if red_center is not None:
        cv2.circle(Mario.frame_with_red_green, red_center, 5, (0, 0, 255), -1)  # Red color for center
        #cv2.putText(Mario.frame_with_red_green, f"Red Area: {red_size}", (red_center[0] - 50, red_center[1] - 20),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    if green_center is not None:
        cv2.circle(Mario.frame_with_red_green, green_center, 5, (0, 255, 0), -1)  # Green color for center
        #cv2.putText(Mario.frame_with_red_green, f"Green Area: {green_size}", (green_center[0] - 50, green_center[1] - 20),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

'''
# initializing and starting multi - thread webcam input stream
if SOURCE != 'webcam':
    webcam_stream = fake_cam('./input.avi')
    print("Using fake cam")

else:
    # initializing and starting multi - thread webcam input stream

    webcam_stream = WebcamStream(stream_id=0)  # 0 id for main camera
    print("Using webcam")
    webcam_stream.start()
# Example usage:
#cap = cv2.VideoCapture(0)  # Open default camera

# Define the color ranges (HSV format)
red_lower1 = np.array([0, 180, 100])
red_upper1 = np.array([10, 255, 255])

red_lower2 = np.array([160, 100, 100])
red_upper2 = np.array([179, 255, 255])


green_lower = np.array([40, 50, 50])
green_upper = np.array([80, 255, 255])
background = Frames_Process.scan_background(webcam_stream)
while True:
    frame = webcam_stream.read()
    mask, mask_4color = Frames_Process.filter_player(frame, background)
    red_mask1 = create_color_mask(frame, red_lower1, red_upper1,mask_4color)
    red_mask2 = create_color_mask(frame, red_lower2, red_upper2,mask_4color)
    # Combine masks for red color
    mask_red = cv2.bitwise_or(red_mask1, red_mask2)
    # Find center of mass and size for red and green colors separately
    red_center, red_size = find_color_mass(mask_red)

    #Green
    green_mask = create_color_mask(frame, green_lower, green_upper,mask_4color)
    green_center, green_size = find_color_mass(green_mask)

    # Draw circles at the center of masses
    if red_center is not None:
        cv2.circle(frame, red_center, 5, (0, 0, 255), -1)  # Red color for center
        cv2.putText(frame, f"Red Area: {red_size}", (red_center[0] - 50, red_center[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    if green_center is not None:
        cv2.circle(frame, green_center, 5, (0, 255, 0), -1)  # Green color for center
        cv2.putText(frame, f"Green Area: {green_size}", (green_center[0] - 50, green_center[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
webcam_stream.vcap.release()
cv2.destroyAllWindows()
'''
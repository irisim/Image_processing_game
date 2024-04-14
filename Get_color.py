import cv2
import numpy as np

def get_color(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv_color = cv2.cvtColor(np.uint8([[frame[y, x]]]), cv2.COLOR_BGR2HSV)
        print("HSV Color:", hsv_color)
'''
# Capture a frame from the video
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if not ret:
    print("Error: Unable to capture frame.")
    exit()

# Create a window to display the frame
cv2.imshow("Frame", frame)
'''
frame = cv2.imread('green_and_red_2.png')
cv2.imshow("Frame", frame)
cv2.setMouseCallback("Frame", get_color)

# Wait for the user to click on the screen
cv2.waitKey(0)

# Release the camera and close all windows
#cap.release()
cv2.destroyAllWindows()

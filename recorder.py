import cv2
import time
import os
import cv2
import numpy as np
# from PIL import Image
# from pygame.locals import *
from threading import Thread,Event

"""______________________________________________________________________________
    *                                                                           *
    *                                                                           *
    *                          camStream setup                                  *
    *                                                                           *
    *___________________________________________________________________________*
"""
OPENCV_LOG_LEVEL = 0
WIDTH, HEIGHT = 1080, 1920
GREEN = (0, 255, 0)
RED = (0, 0, 255)


class FakeCamStream:
    # initialization method
    def __init__(self, source):

        # opening video capture stream
        self.vcap = cv2.VideoCapture(source)
        if self.vcap.isOpened() is False:
            print("[Exiting]: Error accessing webcam stream.")
            exit(0)
        fps_input_stream = int(self.vcap.get(5))  # hardware fps
        print("FPS of input stream: {}".format(fps_input_stream))

        # reading a single frame from vcap stream for initializing
        self.grabbed, self.frame = self.vcap.read()
        self.frame_ready = False
        if self.grabbed is False:
            print('[Exiting] No more frames to read')
            exit(0)
        # self.paused is initialized to False
        self.stopped = True
        self.paused = False

        # thread instantiation
        self.t = Thread(target=self.update, args=())

        self.t.daemon = True  # daemon threads run in background


    ## method to start thread
    def start(self):
        self.stopped = False
        self.t.start()

    def stop(self):
        self.t.join()

    # method passed to thread to read next available frame
    def update(self):
        while True:
            if self.stopped:
                break

            if not self.paused:
                self.grabbed, self.frame = self.vcap.read()
            self.frame_ready = True

            if self.grabbed is False:
                print('[Exiting] No more frames to read')
                self.stopped = True
                break
                # delay for 30 fps
            time.sleep(0.033)

        self.vcap.release()

    # method to return latest read frame
    def read(self):
        return cv2.flip(self.frame, 1)

    # method to stop reading frames
    def pause(self):
        self.paused = not self.paused
        print(f'paused set to {self.paused}⏸')

    def quit(self):
        self.stopped = True
        self.t.join()
        # After the loop release the cap object
        self.vcap.release()
        # Destroy all the windows
        cv2.destroyAllWindows()
        exit(0)

    def get_EXPOSURE(self):

        return 42

    def set_EXPOSURE(self,Exp):

        print("Set current exposure value:", 42, "us")


    # method to play the game with the user's update function




class fake_cam:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
    def read(self):
        _, frame = self.cap.read()
        return frame
    def get_EXPOSURE(self):
        return 42
    def set_EXPOSURE(self, value):
        print(f"Setting exposure to {value}")

if __name__ == '__main__':
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 30.0, (640, 480))

    # Start capturing from the camera
    cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    start_time = time.time()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Write the frame into the file 'input.avi'
        out.write(frame)

        # Show the frame for demo purposes (optional)
        cv2.imshow('frame', frame)

        # Break the loop after 100 seconds
        if time.time() - start_time > 50:
            break

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

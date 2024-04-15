
from keyboard_interface import KeyboardInterface
from recorder import fake_cam, FakeCamStream
from webcam_stream import WebcamStream
import cv2
import numpy as np
import time
#import trash_functions
import Player_Position
import Frames_Process
from Player import Player
import sys
import os
import Player_Control

#SOURCE = 'webcam'    #
SOURCE = 'fake cam'
def play(webcam_stream, background,Mario):
    #Get_player_height
    height_accepted = 0
    while height_accepted != 1:
        Mario.frame = webcam_stream.read()
        # Process the frame
        Mario.mask, Mario.mask_4color = Frames_Process.filter_player(Mario.frame, background)
        Mario.center_of_center, Mario.width, Mario.height_of_person, Mario.percentage = Player_Position.get_player_position(Mario.mask)
        # Display the output
        cv2.imshow('output', Mario.mask)
        # Handle user input
        key = cv2.waitKey(1)
        if key & 0xFF == ord('2'):
            print("Mario.height_of_person = ", Mario.height_of_person,"Mario.center_of_center = ", Mario.center_of_center)
            height_accepted = 1
            break

        if key & 0xFF == ord('3'):
            Mario.height_of_person = 380
            print("Mario.height_of_person = ", Mario.height_of_person,"Mario.center_of_center = ", Mario.center_of_center)
            height_accepted = 1
            break
    print("Press 'b' to edit thresholds\n")
    print("Press 'c' to edit colors\n")
    while True:
        # Capture the video frame
        frame = webcam_stream.read()
        Mario.frame = frame
        Mario.frame_with_red_green = frame.copy()
        # Process the frame
        Mario.mask, Mario.mask_4color = Frames_Process.filter_player(Mario.frame, background)

        Player_Control.player_control(Mario.mask,keyboard,Mario)
        #mask = filter_player(frame, backg1round)
        grid = Frames_Process.grid_output(frame, background,Mario)

        # Display the output
        cv2.imshow('output', grid)

        # Handle user input
        #key = cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(1) & 0xFF == ord('b'):
            Mario.Trashi.alter_threshold()
        elif cv2.waitKey(1) & 0xFF == ord('c'):
            Mario.Colori.frame = Mario.frame
            Mario.Colori.change_color()
        elif key & 0xFF == ord('e'):
            # Assuming get_EXPOSURE is a method of webcam_stream that either prints or sets the exposure
            webcam_stream.get_EXPOSURE()


# initializing and starting multi - thread webcam input stream
if SOURCE != 'webcam':
    webcam_stream = FakeCamStream('./input.avi')
    print("Using fake cam")
    webcam_stream.start()

else:
    # initializing and starting multi - thread webcam input stream

    webcam_stream = WebcamStream(stream_id=0)  # 0 id for main camera
    print("Using webcam")
    webcam_stream.start()
keyboard = KeyboardInterface()
Mario = Player()
background = Frames_Process.scan_background(webcam_stream)
play(webcam_stream, background, Mario)
# After the loop release the cap object
if SOURCE != 'webcam':
    webcam_stream.vcap.release()
else :
    webcam_stream.vcap.release()
# Destroy all the windows
cv2.destroyAllWindows()

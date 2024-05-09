# Import necessary libraries and modules
import math

import numpy as np

from keyboard_interface import KeyboardInterface
from recorder import fake_cam, FakeCamStream
from webcam_stream import WebcamStream
import cv2
import Player_Position
import Frames_Process
from Player import Player
import Player_Control
import os

# Define source of video input
SOURCE = 'input.avi'
#SOURCE = 'webcam'


def play(webcam_stream, background, Mario):
    """
    Main gameplay function handling video processing and user input.

    Args:
        webcam_stream: An instance of a webcam video stream.
        background: Background image for processing.
        Mario: An instance of the Player class.
    """
    height_accepted = 0
    while height_accepted != 1:
        Mario.frame = webcam_stream.read()
        Mario.mask, Mario.mask_4color = Frames_Process.filter_player(Mario.frame, background)
        Mario.center_of_center, Mario.width, Mario.height_of_person, Mario.percentage = Player_Position.get_player_position(
            Mario.mask)

        cv2.imshow('output', Mario.mask)
        key = cv2.waitKey(1)

        if key & 0xFF == ord('2'):
            print(
                f"Mario.height_of_person = {Mario.height_of_person}, Mario.center_of_center = {Mario.center_of_center}")
            height_accepted = 1

        if key & 0xFF == ord('3'):
            Mario.height_of_person = 360  # Set default height if required
            print(
                f"Mario.height_of_person = {Mario.height_of_person}, Mario.center_of_center = {Mario.center_of_center}")
            height_accepted = 1

        if key & 0xFF == ord('q'):
            break

    # Additional user controls
    print("Press 'p' and then 'b' to edit thresholds")
    print("Press 'p' and then 'c' to edit colors")
    Mario.Pause = False
    image_index = 0
    # Define the base directory where folders will be created
    base_dir = 'output'

    # Create the base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    # Construct the folder name and filename
    folder_name_left_T = os.path.join(base_dir, f'Left_leans_true')
    folder_name_right_T = os.path.join(base_dir, f'Right_leans_true')
    folder_name_left_F = os.path.join(base_dir, f'Left_leans_false')
    folder_name_right_F = os.path.join(base_dir, f'Right_leans_false')

    # Create the folder if it doesn't exist
    os.makedirs(folder_name_left_T, exist_ok=True)
    os.makedirs(folder_name_right_T, exist_ok=True)
    os.makedirs(folder_name_left_F, exist_ok=True)
    os.makedirs(folder_name_right_F, exist_ok=True)
    while True:
        while not Mario.Pause:
            frame = webcam_stream.read()
            Mario.frame = frame
            Mario.frame_with_red_green = frame.copy()
            Mario.mask, Mario.mask_4color = Frames_Process.filter_player(Mario.frame, background)

            Player_Control.player_control(Mario.mask, keyboard, Mario)
            grid = Frames_Process.grid_output(frame, background, Mario)
            cv2.imshow('output', grid)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('p'):
                webcam_stream.pause()
                Mario.Pause = True
                break

            if key & 0xFF == ord('q'):
                break

        # Additional user controls
        print("Press 'q' to QUIT")
        print("Press 'l' to resample background")
        print("Press 'b' to edit thresholds")
        print("Press 'c' to edit colors")
        print("Press 'r' to return")
        key = cv2.waitKey(0)
        if key & 0xFF == ord('q'):
            break
        if key & 0xFF == ord('g'):
            filename = f'lean_{image_index}.jpg'  # Construct filename with index i
            image_index = image_index +1
            key2 = cv2.waitKey(0)
            if key2 & 0xFF == ord('1'):
                ## Left
                # Construct the full file path
                filepath = os.path.join(folder_name_left_T, filename)
                # Save the image
                cv2.imwrite(filepath, Mario.frame)
            if key2 & 0xFF == ord('2'):
                ## Right
                # Construct the full file path
                filepath = os.path.join(folder_name_right_T, filename)
                # Save the image
                cv2.imwrite(filepath, Mario.frame)
            if key2 & 0xFF == ord('3'):
                ## Left False
                # Construct the full file path
                filepath = os.path.join(folder_name_left_F, filename)
                # Save the image
                cv2.imwrite(filepath, Mario.frame)
            if key2 & 0xFF == ord('4'):
                ## Right False
                # Construct the full file path
                filepath = os.path.join(folder_name_right_F, filename)
                # Save the image
                cv2.imwrite(filepath, Mario.frame)

            webcam_stream.pause()
            Mario.Pause = False
        elif key & 0xFF == ord('b'):
            Mario.Trashi.alter_threshold()
            webcam_stream.pause()
            Mario.Pause = False
        elif key & 0xFF == ord('c'):
            Mario.Colori.change_color()
            webcam_stream.pause()
            Mario.Pause = False
        elif key & 0xFF == ord('e'):
            webcam_stream.get_EXPOSURE()
            webcam_stream.pause()
            Mario.Pause = False
        elif key & 0xFF == ord('r'):
            webcam_stream.pause()
            Mario.Pause = False
        elif key & 0xFF == ord('l'):
            background = frame.copy()
            webcam_stream.pause()
            Mario.Pause = False




# Initialize and start the webcam input stream
if SOURCE == 'webcam':
    webcam_stream = WebcamStream(stream_id=0)  # 0 is usually the default camera
    print("Using webcam")
    webcam_stream.start()
    webcam_stream.start_recording(output_file='output.avi')
else:
    webcam_stream = FakeCamStream('./input.avi')
    print("Using fake cam")
    webcam_stream.start()

keyboard = KeyboardInterface()
Mario = Player()
background = Frames_Process.scan_background(webcam_stream)
play(webcam_stream, background, Mario)

# Cleanup operations
webcam_stream.stop_recording() if SOURCE == 'webcam' else None
webcam_stream.stop()
webcam_stream.vcap.release()
cv2.destroyAllWindows()

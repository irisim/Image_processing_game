# Import necessary libraries and modules
from keyboard_interface import KeyboardInterface
from recorder import fake_cam, FakeCamStream
from webcam_stream import WebcamStream
import cv2
import Player_Position
import Frames_Process
from Player import Player
import Player_Control

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
            Mario.height_of_person = 380  # Set default height if required
            print(
                f"Mario.height_of_person = {Mario.height_of_person}, Mario.center_of_center = {Mario.center_of_center}")
            height_accepted = 1

        if key & 0xFF == ord('q'):
            break

    # Additional user controls
    print("Press 'b' to edit thresholds\n")
    print("Press 'c' to edit colors\n")

    while True:
        Pausee = False
        while not Pausee :#Mario.Pause:
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
                Pausee = True
                break

            if key & 0xFF == ord('q'):
                break

        key = cv2.waitKey(0)
        if key & 0xFF == ord('q'):
            break
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
        elif key & 0xFF == ord('r'):
            Pausee = False
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

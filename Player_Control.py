import Player_Position
import numpy as np
import time
from pynput.keyboard import Key, Controller, Listener
import Frames_Process

def player_control(mask, keyboard, Mario):
    """
    Control the player based on the detected position and movements.

    Args:
        mask (array): Binary mask indicating player position.
        keyboard (Controller): Instance of the pynput keyboard controller.
        Mario (object): Object containing player information and state.
    """
    # Obtain player position and characteristics
    center_of_mass, width, height, percentage = Player_Position.get_player_position(mask, Mario.Trashi.outlier_std_threshold)
    mask = Player_Position.Region_mask(mask, center_of_mass, height, width)
    center_of_mass, _, _, _ = Player_Position.get_player_position(mask,Mario.Trashi.outlier_std_threshold,only_center=1)
    Mario.mask = mask
    Mario.center_of_mass = center_of_mass
    Mario.width = width
    Mario.height = height
    lean = 'center'
    squat = 0
    # Detect lean, squat, jumps, grabs, and speed
    if not np.isnan(center_of_mass[0]) and not np.isnan(center_of_mass[1]):
        ## iris removed rounding center of mass
        lean, center_of_upper_mass = Player_Position.player_lean(center_of_mass, width, height, w=Mario.width, th=Mario.Trashi.leani, mask=mask)
        Mario.lean, Mario.center_of_upper_mass = lean, center_of_upper_mass

        jumps = Player_Position.jumping(Mario)
        Mario.jump = jumps
        Player_Position.grabing(Mario)
        Player_Position.Faster(Mario)
        Player_Position.slow(Mario)

        if not np.isnan(center_of_upper_mass[0]) and not np.isnan(center_of_upper_mass[1]):
            squat = Player_Position.player_squat(center_of_mass,center_of_upper_mass,th=Mario.Trashi.squati,Height =Mario.height_of_person)
            Mario.squat = squat
            Mario.time_down = time.time()

    # Adjust player controls based on detected movements
    if Mario.stop:
        keyboard.stop_long_press('w')
        keyboard.start_long_press('s')
    else:
        keyboard.stop_long_press('s')

    if Mario.jump == 'up':
        keyboard.stop_long_press(Key.down)
        keyboard.start_long_press(Key.up)
    else:
        keyboard.stop_long_press(Key.up)

    if Mario.faster:
        keyboard.stop_long_press('s')
        keyboard.start_long_press('w')
    else:
        keyboard.stop_long_press('w')

    if Mario.right_grab:
        Mario.time_right_grab = time.time()
        keyboard.start_long_press(Key.right)
    # elif time.time() - Mario.time_right_grab < 1:
    #     keyboard.start_long_press(Key.right) ### If you used left grab in the last sec (before jumping) consider it
    else:
        keyboard.stop_long_press(Key.right)

    if Mario.left_grab:
        Mario.time_left_grab = time.time()
        keyboard.start_long_press(Key.left)
    # elif time.time() - Mario.time_left_grab < 1:
    #     keyboard.start_long_press(Key.left) ### If you used left grab in the last sec (before jumping) consider it
    else:
        keyboard.stop_long_press(Key.left)

    if squat == 'down':
        Mario.time_down = time.time()
        keyboard.start_long_press(Key.down)
    else:
        keyboard.stop_long_press(Key.down)

    if lean == 'left':
        keyboard.stop_long_press("d")
        keyboard.start_long_press("a")
    elif lean == 'right':
        keyboard.stop_long_press("a")
        keyboard.start_long_press("d")
    else:
        keyboard.stop_long_press("a")
        keyboard.stop_long_press("d")

    Mario.previous_mask = mask
    Mario.last_center = center_of_mass


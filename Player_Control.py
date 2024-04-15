import Player_Position
import numpy as np
import time
from pynput.keyboard import Key, Controller, Listener
import Frames_Process

def player_control(mask,keyboard, Mario):

    #center_of_mass, width, height = Mario.center_of_mass , Mario.width , Mario.height
    lean = 'center'
    squat = 0
    ########dad
    center_of_mass, width, height, percentage = Player_Position.get_player_position(mask, Mario.Trashi.outlier_std_threshold)
    mask = Player_Position.Region_mask(mask, center_of_mass, height, width)
    Mario.mask = mask
    Mario.center_of_mass = center_of_mass
    Mario.width = width
    Mario.height = height
    #########
    if not np.isnan(center_of_mass[0]) and not np.isnan(center_of_mass[1]):
        center_of_mass = (round(center_of_mass[0]), round(center_of_mass[1]))
        #Mario.center_of_mass = center_of_mass
        lean, center_of_upper_mass = Player_Position.player_lean(center_of_mass,width, height,w=Mario.width,th = Mario.Trashi.leani, mask=mask)
        Mario.lean = lean
        Mario.center_of_upper_mass = center_of_upper_mass
        jumps = Player_Position.jumping(Mario)
        Mario.jump = jumps
        Player_Position.grabing(Mario)
        if not np.isnan(center_of_upper_mass[0]) and not np.isnan(center_of_upper_mass[1]):
            #squat = player_squat(center_of_mass,center_of_upper_mass,th=1,H=Mario.H)
            squat = Player_Position.player_squat(center_of_mass,center_of_upper_mass,th=Mario.Trashi.squati,Height =Mario.height_of_person)
            Mario.squat = squat
            #### Is it neccessary here ? yes
            #center_of_upper_mass = (round(center_of_upper_mass[0]), round(center_of_upper_mass[1]))
            #Mario.center_of_upper_mass = center_of_upper_mass

    #print(lean)
    if Mario.jump == 'up':
        print("up")
        keyboard.stop_long_press(Key.down)
        keyboard.start_long_press(Key.up)

    if time.time() - Mario.time_right_grab > 2:
        keyboard.stop_long_press(Key.right)
    if Mario.right_grab == True:
        print("right grab")
        ###keyboard.stop_long_press(Key.left) DO NOT USE
        Mario.time_right_grab = time.time()
        keyboard.start_long_press(Key.right)

    if time.time() - Mario.time_left_grab > 2:
        keyboard.stop_long_press(Key.left)
    if Mario.left_grab == True:
        print("left grab")
        ####keyboard.stop_long_press(Key.right) DO NOT USE
        Mario.time_left_grab = time.time()
        keyboard.start_long_press(Key.left)

    if squat == 'down':
        print("down")
        Mario.set_down()
        keyboard.start_long_press(Key.down)
    if squat == 0:
        keyboard.stop_long_press(Key.down)
    if lean == 'left':
        print("left")
        keyboard.stop_long_press("d")
        keyboard.start_long_press("a")

    if lean == 'right':
        print("right")
        keyboard.stop_long_press("a")
        keyboard.start_long_press("d")
    if Mario.right_grab != True:
        keyboard.stop_long_press(Key.right)
    if Mario.left_grab != True:
        keyboard.stop_long_press(Key.left)
    if lean == 'center':
        #print("center")
        keyboard.stop_long_press("a")
        keyboard.stop_long_press("d")

    '''    if Mario.jump == 'up':
        print("up")
        keyboard.press_and_release(Key.up)
    if squat == 'down':
        print("down")
        Mario.set_down()
        keyboard.press_and_release(Key.down)
    if lean == 'left':
        print("left")
        keyboard.press_and_release('a')
    if lean == 'right':
        print("right")
        keyboard.press_and_release('d')'''

    Mario.previous_mask = mask
    Mario.last_center = Mario.center_of_mass

    # add controls here
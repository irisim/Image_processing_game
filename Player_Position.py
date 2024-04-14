
from webcam_stream import WebcamStream
import cv2
import numpy as np
import time
from scipy import ndimage
from pynput.keyboard import Key, Controller, Listener
import Frames_Process
from Player import Player
import colors_process

Mario = Player()
def get_player_position(mask,outlier_std_threshold=5):
    """
    :param mask: binary mask
    :return: (center_x, center_y)
    """
    # Find indices where we have mass
    mass_h, mass_w = np.where(mask == 255)

    # x,y are the center of x indices and y indices of mass pixels
    center_of_mass = (np.average(mass_w), np.average(mass_h))

    if len(mass_w) < 10 or len(mass_h) < 10:
        #center_of_mass = (mask.shape[0]//2,mask.shape[1]//2)
        return center_of_mass, 10, 10, 0

    # Calculate distances of each pixel from the center of mass
    distances = np.sqrt((mass_h - center_of_mass[1]) ** 2 + (mass_w - center_of_mass[0]) ** 2)

    # Filter out outliers based on the standard deviation threshold
    std_dev = np.std(distances)
    main_mass_indices = np.where(distances <= outlier_std_threshold * std_dev)

    # Use only main mass indices to calculate width and height
    main_mass_w = mass_w[main_mass_indices]
    main_mass_h = mass_h[main_mass_indices]

    # Calculate percentage of mask pixels being equal to 1
    total_pixels = mask.shape[0] * mask.shape[1]
    ones_count = np.count_nonzero(mask)
    percentage = (ones_count / total_pixels) * 100

    # Calculate width and height of the rectangle
    if len(main_mass_w) < 50 or len(main_mass_h) < 50:
        width = 50
        height = 50
        return center_of_mass, width, height, percentage
    width = np.max(main_mass_w) - np.min(main_mass_w)
    height = np.max(main_mass_h) - np.min(main_mass_h)

    if not np.isnan(center_of_mass[0]) and not np.isnan(center_of_mass[1]):
        center_of_mass = (round(center_of_mass[0]), round(center_of_mass[1])) ## cancel if you want the accuracy
    return center_of_mass, width, height, percentage

def player_squat(center_of_mass,center_of_upper_mass,th=1,Height = 300):
    delta = Height// th
    y = center_of_mass[1]

    if center_of_upper_mass[1] > y - delta:
        #print("center_of_upper_mass[0]-y=", center_of_upper_mass[0]-y)
        #print("center_of_upper_mass[0]=", center_of_upper_mass[0], "y=", y, "y-th=", y - th, "th=", th)
        return 'down'
    return 0
def player_lean(center_of_mass,width, height, w = 640 , th = 4,mask = None):
    # calculate the threshold precentage
    #print("W=",w)
    th = w*th//100
    # width
    x = center_of_mass[0]

    # height
    y = center_of_mass[1]
    # operate in the region of the player's upper body
    if mask is None:
        return
    # TODO: make this work:

    mask_region = Upper_Region_mask(mask,center_of_mass,height,width)

    uppermass_h, uppermass_w = np.where(mask_region == 255)

    # x,y are the center of x indices and y indices of mass pixels
    center_of_upper_mass = (np.average(uppermass_w), np.average(uppermass_h))
    if center_of_upper_mass[0] > x + th:
        return 'right',center_of_upper_mass
    if center_of_upper_mass[0] < x - th:
        return 'left',center_of_upper_mass
    return 'center', center_of_upper_mass

def jumping(Mario):
    if (time.time() - Mario.time_down > 2 and time.time() - Mario.time_up > 0.5) :
        if ( Mario.last_center[1] - Mario.center_of_mass[1] > 4 ):
            Mario.time_up = time.time()
            return 'up'
    else:
        return 'center'

def grabing(Mario):
    Time = time.time()
    #if (Time - Mario.time_up < 7 or Time - Mario.time_down < 7) :
    colors_process.get_green_and_red(Mario)
    green_location = Mario.green_center
    red_location = Mario.red_center
    Frames_Process.draw_spot_info(Mario.frame_with_red_green, green_location, "green")
    Frames_Process.draw_spot_info(Mario.frame_with_red_green, red_location, "red")
    bottom_height = Mario.center_of_mass[1] + Mario.height//2
    limit_bottom_height = Mario.center_of_mass[1] + Mario.height//(2*5)
    if green_location != None :
        if limit_bottom_height < green_location[1]: # or limit_bottom_height < red_location[1] :
            print("Green grab, green_location =",green_location )
            print("bottom_height =",bottom_height ,"limit_bottom_height =",limit_bottom_height )
            Mario.right_grab = True
        else :
            Mario.right_grab = False


    if red_location != None :
        if limit_bottom_height < red_location[1] :
            print("Red Grab, red_location =",red_location )
            print("bottom_height =",bottom_height ,"limit_bottom_height =",limit_bottom_height )
            Mario.left_grab = True
        else :
            Mario.left_grab = False



def player_control(mask,keyboard, Mario):

    #center_of_mass, width, height = Mario.center_of_mass , Mario.width , Mario.height
    lean = 'center'
    squat = 0
    ########dad
    center_of_mass, width, height, percentage = get_player_position(mask)
    mask = Region_mask(mask, center_of_mass, height, width)
    Mario.mask = mask
    Mario.center_of_mass = center_of_mass
    Mario.width = width
    Mario.height = height
    #########
    if not np.isnan(center_of_mass[0]) and not np.isnan(center_of_mass[1]):
        center_of_mass = (round(center_of_mass[0]), round(center_of_mass[1]))
        Mario.center_of_mass = center_of_mass
        lean, center_of_upper_mass = player_lean(center_of_mass,width, height,w=Mario.width, mask=mask)
        Mario.lean = lean
        Mario.center_of_upper_mass = center_of_upper_mass
        jumps = jumping(Mario)
        Mario.jump = jumps
        grabing(Mario)
        if not np.isnan(center_of_upper_mass[0]) and not np.isnan(center_of_upper_mass[1]):
            #squat = player_squat(center_of_mass,center_of_upper_mass,th=1,H=Mario.H)
            squat = player_squat(center_of_mass,center_of_upper_mass,th=Mario.squat_th,Height =Mario.height_of_person)
            Mario.squat = squat
            #### Is it neccessary here ? yes
            center_of_upper_mass = (round(center_of_upper_mass[0]), round(center_of_upper_mass[1]))
            Mario.center_of_upper_mass = center_of_upper_mass

    #print(lean)
    if Mario.jump == 'up':
        print("up")
        keyboard.stop_long_press(Key.down)
        keyboard.start_long_press(Key.up)

    if Mario.time_right_grab < 2:
        keyboard.start_long_press(Key.right)
    if Mario.right_grab == True:
        print("right grab")
        ###keyboard.stop_long_press(Key.left) DO NOT USE
        Mario.time_right_grab = time.time()
        keyboard.start_long_press(Key.right)

    if Mario.time_left_grab < 2:
        keyboard.start_long_press(Key.left)
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


def Upper_Region_mask(mask,center_of_mass,height,width):
    mask_region = np.zeros_like(mask)
    if not np.isnan(center_of_mass[0]) and not np.isnan(center_of_mass[1]):
        center_of_mass = (round(center_of_mass[0]), round(center_of_mass[1]))
        # Calculate the rectangle boundaries
        top = max(center_of_mass[1] - height // 2, 0)  # Ensure top is not less than 0
        bottom = min(center_of_mass[1], mask.shape[0])  # Ensure bottom does not exceed mask height
        left = max(center_of_mass[0] - width // 2, 0)  # Ensure left is not less than 0
        right = min(center_of_mass[0] + width // 2, mask.shape[1])  # Ensure right does not exceed mask width

        # Replace the region within the bounds with the corresponding values from the original mask
        mask_region[top:bottom, left:right] = mask[top:bottom, left:right]
    # iris
    return mask_region

def Region_mask(mask,center_of_mass,height,width):
    mask_region = np.zeros_like(mask)
    if not np.isnan(center_of_mass[0]) and not np.isnan(center_of_mass[1]):
        center_of_mass = (round(center_of_mass[0]), round(center_of_mass[1]))
        # Calculate the rectangle boundaries
        top = max(center_of_mass[1] - height // 2, 0)  # Ensure top is not less than 0
        bottom = min(center_of_mass[1] + height // 2, mask.shape[0])
        left = max(center_of_mass[0] - width // 2, 0)  # Ensure left is not less than 0
        right = min(center_of_mass[0] + width // 2, mask.shape[1])  # Ensure right does not exceed mask width

        # Replace the region within the bounds with the corresponding values from the original mask
        mask_region[top:bottom, left:right] = mask[top:bottom, left:right]
    # iris
    return mask_region

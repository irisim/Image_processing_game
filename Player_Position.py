
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

    return center_of_mass, width, height, percentage

def jumping(Mario, th = 4):
    if (time.time() - Mario.time_down > 0.5 and time.time() - Mario.time_up > 0.3) :
        if ( Mario.last_center[1] - Mario.center_of_mass[1] > (Mario.Trashi.jumpi/100) ):
            Mario.time_up = time.time()
            return 'up'
    else:
        return 'center'

def player_squat(center_of_mass,center_of_upper_mass,th=100,Height = 300):
    delta = Height*100/ th
    y = center_of_mass[1]

    if center_of_upper_mass[1] > y - delta:
        #print("center_of_upper_mass[0]-y=", center_of_upper_mass[0]-y)
        #print("center_of_upper_mass[0]=", center_of_upper_mass[0], "y=", y, "y-th=", y - th, "th=", th)
        return 'down'
    return 0
def player_lean(center_of_mass,width, height, w = 300 , th = 10,mask = None):
    # calculate the threshold precentage
    #print("W=",w)ddd
    delta = th ##w*th//10000
    # width
    x = center_of_mass[0]

    # height
    y = center_of_mass[1]
    # operate in the region of the player's upper body
    if mask is None:
        return
    # TODO: make this work:

    mask_region = Upper_Region_mask(mask,center_of_mass,height,width)
    Lower = Lowwer_Region_mask(mask, center_of_mass, height, width)
    Lowwer_h, Lowwer_w = np.where(Lower == 255)
    uppermass_h, uppermass_w = np.where(mask_region == 255)

    # x,y are the center of x indices and y indices of mass pixels
    center_of_upper_mass = (np.average(uppermass_w), np.average(uppermass_h))
    center_of_lowwer_mass = (np.average(Lowwer_w), np.average(Lowwer_h))
    if center_of_upper_mass[0] > center_of_lowwer_mass[0] + delta:
        return 'right',center_of_upper_mass
    if center_of_upper_mass[0] < center_of_lowwer_mass[0] - delta:
        return 'left',center_of_upper_mass
    return 'center', center_of_upper_mass


def grabing(Mario):
    Time = time.time()
    #if (Time - Mario.time_up < 7 or Time - Mario.time_down < 7) :
    colors_process.get_green_and_red(Mario)
    green_location = Mario.green_center
    red_location = Mario.red_center
    Frames_Process.draw_spot_info(Mario.frame_with_red_green, green_location, "green")
    Frames_Process.draw_spot_info(Mario.frame_with_red_green, red_location, "red")
    bottom_height = Mario.center_of_mass[1] + Mario.height//2
    limit_bottom_height = Mario.center_of_mass[1] + Mario.height_of_person//(2*Mario.Trashi.grabi/100)
    if green_location != None :
        if limit_bottom_height < green_location[1]: # or limit_bottom_height < red_location[1] :
            print("Green grab, green_location =",green_location )
            print("bottom_height =",bottom_height ,"limit_bottom_height =",limit_bottom_height )
            Mario.right_grab = True
            Mario.time_right_grab = time.time()
        else :
            Mario.right_grab = False


    if red_location != None :
        if limit_bottom_height < red_location[1] :
            print("Red Grab, red_location =",red_location )
            print("bottom_height =",bottom_height ,"limit_bottom_height =",limit_bottom_height )
            Mario.left_grab = True
            Mario.time_left_grab = time.time()
        else :
            Mario.left_grab = False


def faster(Mario):
    green_location = Mario.green_center
    red_location = Mario.red_center
    ##[0]=w, ## [1]=h
    if red_location != None and green_location != None :
        if (np.abs(green_location[1] - red_location[1]) < 20) and np.abs(green_location[0] - red_location[0]) > 150 :
            Mario.faster = True
        else :
            Mario.faster = False
    else:
        Mario.faster = False


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

def Lowwer_Region_mask(mask,center_of_mass,height,width):
    mask_region = np.zeros_like(mask)
    if not np.isnan(center_of_mass[0]) and not np.isnan(center_of_mass[1]):
        center_of_mass = (round(center_of_mass[0]), round(center_of_mass[1]))
        # Calculate the rectangle boundaries
        top = max(center_of_mass[1], 0)  # Ensure top is not less than 0
        bottom = min(center_of_mass[1] + height // 2, mask.shape[0])  # Ensure bottom does not exceed mask height
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

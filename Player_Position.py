
import cv2
import numpy as np
import time
import Frames_Process
from Player import Player
import colors_process
import math

Mario = Player()
def get_player_position(mask,outlier_std_threshold=5,only_center = 0):
    """
    :param mask: binary mask
    :return: (center_x, center_y)
    """
    # Find indices where we have mass
    mass_h, mass_w = np.where(mask == 255)

    # x,y are the center of x indices and y indices of mass pixels
    center_of_mass = (np.average(mass_w), np.average(mass_h))
    if only_center == 1:
        return center_of_mass, 10, 10, 0
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
    if (time.time() - Mario.time_down > 0.05  ): #and time.time() - Mario.time_up > 0.3) :
        if ( Mario.last_center[1] - Mario.center_of_mass[1] > (Mario.Trashi.jumpi/100) ):
            Mario.time_up = time.time()
            return 'up'
    else:
        return 'center'

def player_squat(center_of_mass,center_of_upper_mass,th=100,Height = 300):
    if (time.time() - Mario.time_up > 0.05):
        delta = Height*100/ th
        y = center_of_mass[1]

        if center_of_upper_mass[1] > y - delta:
            return 'down'
        return 0
def player_lean(center_of_mass,width, height, w = 300 , th = 10,mask = None):
    delta = th
    # width
    x = center_of_mass[0]
    # height
    y = center_of_mass[1]
    # operate in the region of the player's upper body
    if mask is None:
        return

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
    # if (Time - Mario.time_up < 7 or Time - Mario.time_down < 7) :
    colors_process.get_green_and_red(Mario)
    green_location = Mario.green_center
    red_location = Mario.red_center
    Frames_Process.draw_spot_info(Mario.frame_with_red_green, green_location, "green")
    Frames_Process.draw_spot_info(Mario.frame_with_red_green, red_location, "red")
    grab_thresh_y = Mario.center_of_mass[1] + Mario.height_of_person // (2 * Mario.Trashi.grabi / 100)
    #grab_thresh_x = Mario.center_of_mass[0] + Mario.width // (2 * Mario.Trashi.grab_x / 100)
    grabbed = False

    if green_location is not None:
        if grab_thresh_y < green_location[1]:       # Height Threshold achieved, need to grab right or left
            grabbed = True      # There is a grab using green, don't check for red
            if Mario.center_of_mass[0] < green_location[0]:   # Grab right
                Mario.left_grab = False
                Mario.right_grab = True
                Mario.time_right_grab = time.time()
            else:       # Grab left
                Mario.right_grab = False
                Mario.left_grab = True
                Mario.time_right_grab = time.time()
        else:
            Mario.left_grab = False
            Mario.right_grab = False

    if red_location is not None and grabbed is not True:
        if grab_thresh_y < red_location[1]:       # Height Threshold achieved, need to grab right or left
            if Mario.center_of_mass[0] < red_location[0]:   # Grab right
                Mario.left_grab = False
                Mario.right_grab = True
                Mario.time_right_grab = time.time()
            else:       # Grab left
                Mario.right_grab = False
                Mario.left_grab = True
                Mario.time_right_grab = time.time()
        else:
            Mario.left_grab = False
            Mario.right_grab = False

def Faster(Mario):
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

def slow(Mario):
    """
    Process the given 'Mario' object to detect specific leg positions
    and decide whether to stop based on line intersections in the image.

    Args:
        Mario (object): An object that contains image data and state information.
    """
    # Copy the mask of the Mario object and zero out the top half (presumably to ignore it).
    mask = Mario.mask.copy()
    mask[:240, :] = 0

    # Initialize an image for drawing lines detected.
    mask_lines = np.zeros_like(Mario.frame)
    added_line = np.zeros_like(Mario.frame)
    Mario.mask_lines = mask_lines

    # Detect edges in the mask.
    edges = cv2.Canny(mask, 100, 200)

    # Apply Hough Transform to detect lines.
    lines = cv2.HoughLines(edges, rho=4, theta=np.pi/180, threshold=80)
    right_leg, left_leg = False, False
    right_x0_y0, left_x0_y0 = [], []

    # Process each detected line.
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            degrees_value = math.degrees(theta)

            # Check angles for right and left leg assumptions.
            if 20 < abs(degrees_value) < 40:
                right_leg = True
                x1, y1, x2, y2 = calculate_line_endpoints(rho, theta)
                cv2.line(added_line, (x1, y1), (x2, y2), (255, 0, 10), 2)
                mask_lines += added_line
                right_x0_y0.append((rho * np.cos(theta), rho * np.sin(theta)))

            if 140 < abs(degrees_value) < 160:
                left_leg = True
                x1, y1, x2, y2 = calculate_line_endpoints(rho, theta)
                cv2.line(added_line, (x1, y1), (x2, y2), (255, 0, 10), 2)
                mask_lines += added_line
                left_x0_y0.append((rho * np.cos(theta), rho * np.sin(theta)))

    # Update mask lines in Mario object.
    Mario.mask_lines = mask_lines

    # Check for stop conditions if both right and left legs detected.
    if right_leg and left_leg:
        check_for_stop(Mario, right_x0_y0, left_x0_y0, mask_lines)
    else:
        Mario.stop = False

def calculate_line_endpoints(rho, theta):
    """
    Calculate line endpoints for drawing based on rho and theta of the line.

    Args:
        rho (float): The distance from the origin to the line.
        theta (float): The angle of rotation from the x-axis.

    Returns:
        tuple: The coordinates (x1, y1, x2, y2) of the line endpoints.
    """
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    return x1, y1, x2, y2

def check_for_stop(Mario, right_x0_y0, left_x0_y0, mask_lines):
    """
    Check if Mario should stop based on the positions of detected lines.

    Args:
        Mario (object): Mario object with mask data.
        right_x0_y0 (list): List of coordinates for right leg lines.
        left_x0_y0 (list): List of coordinates for left leg lines.
        mask_lines (array): Image array where lines are drawn.
    """
    for right_x0, right_y0 in right_x0_y0:
        for left_x0, left_y0 in left_x0_y0:
            if abs(right_x0 - left_x0) < 125 and 120 > abs(left_y0 - right_y0) > 55:
                intersection = np.where(mask_lines[:, :, 2] > 11)  # Check intersections in the red channel.
                print("intersection[0]=",intersection[0])
                if len(intersection[0]) > 0 and any(240 <= y <= 370 for y in intersection[0]):
                    Mario.stop = True
                    Mario.mask_lines = mask_lines
                    print(f"Intersection detected between {right_x0, right_y0} and {left_x0, left_y0}. STOPPPPPPPPPPPPPPPP")






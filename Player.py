from webcam_stream import WebcamStream
import cv2
import numpy as np
import time
from scipy import ndimage
from pynput.keyboard import Key, Controller, Listener
import Frames_Process
import Player_Position
import Get_color
import Thresholds

class Player:
    def __init__(self):
        self.frame = None
        self.previous_mask = None
        self.current_mask = None
        self.center_of_mass = None
        self.last_center = (0,0)
        self.height = None
        self.width = None
        self.height_of_person = None
        self.squat_th = 7 ## Smaller -> more squats easy, larger -> less squatsd
        self.center_of_upper_mass = None
        self.upper_mask = None
        self.H = 480
        self.W = 640
        self.lean = None
        self.squat = None
        self.jump = None
        self.time_down = time.time()
        self.time_up = time.time()
        self.center_of_center = None
        ## X -> Width -> [0], Y -> Height -> [1]
        self.mask_4color = None
        self.green_center, self.green_size = None, None
        self.red_center, self.red_size = None, None
        self.frame_with_red_green = None
        self.right_grab = None
        self.left_grab = None
        self.time_right_grab = time.time()
        self.time_left_grab = time.time()
        self.Trashi = Thresholds.Trashi()
        self.Colori = Get_color.Colori()
        self.pause = False
        self.faster = False

    def set_down(self):
        self.time_down = time.time()

    '''    def set_jump(self,jumping):
        Times = time.time() - self.time_down
        if Times > 3 :
            print("time.time() - self.time_down = ", Times)
            self.jump = jumping
        else:
            print("Times = ", Times)
            self.jump = None'''






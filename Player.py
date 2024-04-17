
import time
import Get_color
import Thresholds

class Player:
    def __init__(self):
        ### Updated movments:
        self.lean = None
        self.squat = None
        self.jump = None
        self.right_grab = None
        self.left_grab = None
        self.faster = False
        self.stop = False
        self.Pause = False

        ### Timing info for made movments:
        self.time_right_grab = time.time()
        self.time_left_grab = time.time()
        self.time_down = time.time()
        self.time_up = time.time()

        ### Frames and masks info:
        self.frame = None
        self.previous_mask = None
        self.current_mask = None
        self.upper_mask = None
        self.mask_4color = None
        self.frame_with_red_green = None
        self.mask_lines = None

        ### Extracted info like locations:
        ## X -> Width -> [0], Y -> Height -> [1]
        self.center_of_mass = None
        self.center_of_upper_mass = None
        self.green_center, self.green_size = None, None
        self.red_center, self.red_size = None, None
        self.last_center = (0,0)
        self.height = None
        self.width = None

        ### Persons initial info:
        self.height_of_person = None
        self.H = 480
        self.W = 640
        self.center_of_center = None

        ### Thresholds and color range
        self.Trashi = Thresholds.Trashi()
        self.Colori = Get_color.Colori()











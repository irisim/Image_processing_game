import cv2
import time
from threading import Thread


class WebcamStream:
    def __init__(self, stream_id=0):
        self.stream_id = stream_id
        self.vcap = cv2.VideoCapture(self.stream_id)
        if not self.vcap.isOpened():
            print("[Error]: Unable to access webcam stream.")
            exit(0)
        fps_input_stream = int(self.vcap.get(cv2.CAP_PROP_FPS))
        print("FPS of input stream: {}".format(fps_input_stream))
        self.grabbed, self.frame = self.vcap.read()
        self.EXP_VALUE = -6
        self.frame_ready = False
        self.stopped = True
        self.paused = False
        self.recording = False
        self.out = None
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True

    def start(self):
        self.stopped = False
        self.t.start()

    def update(self):
        while True:
            if self.stopped:
                break
            self.grabbed, self.frame = self.vcap.read()
            self.frame_ready = True
            if self.recording:
                if self.out is None:
                    self.start_recording()
                self.out.write(self.frame)
            if not self.grabbed:
                print('[Exiting] No more frames to read')
                self.stopped = True
                break
            time.sleep(0.033)
        self.vcap.release()

    def read(self):
        return cv2.flip(self.frame, 1)

    def pause(self):
        self.paused = not self.paused
        print('paused⏸' * self.paused + f'resumed⏯' * (not self.paused))

    def stop(self):
        self.stopped = True

    def start_recording(self, output_file='output.avi', codec='XVID', fps=30.0, resolution=(640, 480)):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.out = cv2.VideoWriter(output_file, fourcc, fps, resolution)
        self.recording = True

    def stop_recording(self):
        if self.out is not None:
            self.out.release()
            self.out = None
            self.recording = False

    def quit(self):
        self.stop()
        self.stop_recording()
        self.t.join()
        cv2.destroyAllWindows()
        exit(0)

    def get_EXPOSURE(self):
        exposure_value_us = self.vcap.get(cv2.CAP_PROP_EXPOSURE)
        print("Current exposure value:", exposure_value_us, "us")
        return exposure_value_us

    def set_EXPOSURE(self, exposure_value_us):
        self.vcap.set(cv2.CAP_PROP_EXPOSURE, exposure_value_us)
        print("Set current exposure value:", exposure_value_us, "us")

    # Increase the EXPOSURE VALUE by 1
    def increase_exposure(self):
        self.EXP_VALUE += 1
        self.vcap.set(cv2.CAP_PROP_EXPOSURE, self.EXP_VALUE)
        print(f'Exposure Value increased by 1, currently its {self.EXP_VALUE}\n')

    # Decrease the EXPOSURE VALUE by 1
    def decrease_exposure(self):
        self.EXP_VALUE -= 1
        self.vcap.set(cv2.CAP_PROP_EXPOSURE, self.EXP_VALUE)
        print(f'Exposure Value increased by 1, currently its {self.EXP_VALUE}\n')

    # Increase the EXPOSURE VALUE by 1
    def increase_exposure(self):
        self.EXP_VALUE += 1
        self.vcap.set(cv2.CAP_PROP_EXPOSURE, self.EXP_VALUE)
        print(f'Exposure Value increased by 1, currently its {self.EXP_VALUE}\n')

    # Decrease the EXPOSURE VALUE by 1
    def decrease_exposure(self):
        self.EXP_VALUE -= 1
        self.vcap.set(cv2.CAP_PROP_EXPOSURE, self.EXP_VALUE)
        print(f'Exposure Value increased by 1, currently its {self.EXP_VALUE}\n')

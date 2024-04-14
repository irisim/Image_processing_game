import cv2
import time



class fake_cam:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
    def read(self):
        _, frame = self.cap.read()
        return frame
    def get_EXPOSURE(self):
        return 42
    def set_EXPOSURE(self, value):
        print(f"Setting exposure to {value}")

if __name__ == '__main__':
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 30.0, (640, 480))

    # Start capturing from the camera
    cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    start_time = time.time()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Write the frame into the file 'input.avi'
        out.write(frame)

        # Show the frame for demo purposes (optional)
        cv2.imshow('frame', frame)

        # Break the loop after 100 seconds
        if time.time() - start_time > 50:
            break

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

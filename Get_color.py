import numpy as np
import cv2

class Colori:
    def __init__(self):
        # Define the color ranges (HSV format)
        ## RED
        self.red_lower1 = np.array([0, 180, 100])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([160, 100, 100])
        self.red_upper2 = np.array([179, 255, 255])
        ## GREEN
        self.green_lower = np.array([40, 50, 50])
        self.green_upper = np.array([80, 255, 255])

        self.frame = cv2.imread("green_and_red_2.png")  # Frame to be used for color picking

    def verify_int_array(self, prompt, size=3):
        while True:
            values = input(prompt).strip().split()
            if all(v.isdigit() for v in values) and len(values) == size:
                return np.array([int(v) for v in values])
            else:
                print("Invalid input. Please enter three integers separated by spaces.")

    def display_menu(self):
        print("Select the color range to alter:")
        print("1 - Red Lower Bound 1")
        print("2 - Red Upper Bound 1")
        print("3 - Red Lower Bound 2")
        print("4 - Red Upper Bound 2")
        print("5 - Green Lower Bound")
        print("6 - Green Upper Bound")
        print("8 - Touch and get HSV Color from the screen")
        print("7 - Exit")

    def touch_color(self):
        cv2.imshow("Frame", self.frame)
        cv2.setMouseCallback("Frame", self.get_color)
        # Wait for the user to click on the screen
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == 27:  # ESC key
                print("Exiting color touch mode...")
                break
        cv2.destroyWindow("Frame")

    def get_color(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            hsv_color = cv2.cvtColor(self.frame[y:y+1, x:x+1], cv2.COLOR_BGR2HSV)
            print("HSV Color:", hsv_color[0][0])

    def change_color(self):
        #self.frame = cv2.imread("green_and_red_2.png")  # Ensure to load an appropriate image or set it externally
        while True:
            self.display_menu()
            choice = input("Enter your choice: ")
            if choice.isdigit():
                choice = int(choice)
                if choice == 7:
                    print("Exiting...")
                    break
                elif choice == 8:
                    self.touch_color()
                    continue

                threshold_name = self.get_attribute_name(choice)
                current_value = getattr(self, threshold_name)
                print(f"Current value for {threshold_name}: {current_value}")

                new_value = self.verify_int_array(f"Enter new values for {threshold_name} (H S V): ")
                setattr(self, threshold_name, new_value)
                print("Value updated successfully!")

    def get_attribute_name(self, choice):
        mapping = {
            1: 'red_lower1',
            2: 'red_upper1',
            3: 'red_lower2',
            4: 'red_upper2',
            5: 'green_lower',
            6: 'green_upper'
        }
        return mapping.get(choice, 'invalid')

# Example usage
if __name__ == "__main__":
    colori = Colori()
    colori.change_color()




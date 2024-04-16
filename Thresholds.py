import sys

class Trashi:
    def __init__(self):
        self.squati = 600  # Smaller -> more squats easy, larger -> less squats, range [600,700] 650
        self.leani = 20   # Smaller -> more action easily, larger -> less actions
        self.grabi = 500   # Smaller -> LESS action easily, larger -> MORE actions
        self.jumpi = 400   # Smaller -> more action easily, larger -> less actions
        self.outlier_std_threshold = 5

    def verify_int(self, prompt):
        while True:
            choice = input(prompt)
            if choice.isdigit():
                print("Input is an integer.")
                return int(choice)
            else:
                print("Input is not an integer. Please enter a valid integer.")

    def display_menu(self):
        print("Select the threshold to alter:")
        print("1 - Squat Threshold")
        print("2 - Lean Threshold")
        print("3 - Grab Threshold")
        print("4 - Jump Threshold")
        print("5 - Outlier Standadrd Deviation Threshold")
        print("6 - Exit")

    def alter_threshold(self):
        while True:
            self.display_menu()
            choice = self.verify_int("Enter your choice: ")
            if choice == 6:
                print("Exiting...")
                break

            threshold_name = self.get_attribute_name(choice)
            current_value = getattr(self, threshold_name)
            print(f"Current value for selected threshold: {current_value}")

            # Include the threshold name in the prompt for the new value
            new_value = self.verify_int(f"Enter new value for {threshold_name.replace('_', ' ').capitalize()}: ")
            setattr(self, self.get_attribute_name(choice), new_value)
            print("Value updated successfully!")
            answer = input("Would you like to update another value ? y/n : ")
            if answer == 'y' :
                continue
            else:
                break

    def get_attribute_name(self, choice):
        mapping = {
            1: 'squati',
            2: 'leani',
            3: 'grabi',
            4: 'jumpi',
            5: 'outlier_std_threshold'
        }
        return mapping.get(choice, 'invalid')

# Example usage
if __name__ == "__main__":
    trashi = Trashi()
    trashi.alter_threshold()

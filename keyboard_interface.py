from pynput.keyboard import Key, Controller, Listener
import threading
import time

class KeyboardInterface:
    def __init__(self):
        self.keyboard = Controller()
        self.auto_press_flags = {}
        self.long_press_flags = {}
        self.soft_press_flags = {}
        self.threads = {}
        self.press_flags ={}
        self.user_only_mode = False
    def start_long_press(self, key):
        """
        Starts a long press on a specific key.

        :param key: The key to start pressing.
        """
        def press_key():
            self.keyboard.press(key)
            while self.long_press_flags.get(key, False):  # Continue pressing the key as long as the flag is True
                self.long_press_flags[key] = True  # Set the flag to True to indicate the key is being pressed
                time.sleep(0.1)  # Sleep briefly to avoid CPU overuse
            self.keyboard.release(key)  # Ensure the key is released when done

        if key not in self.long_press_flags or not self.long_press_flags[key]:
            self.long_press_flags[key] = True  # Set the flag to True to indicate the key is being pressed
            thread = threading.Thread(target=press_key)
            thread.start()
            self.threads[key] = thread  # Store the thread for possible termination

    def stop_long_press(self, key):
        """
        Stops a long press on a specific key.

        :param key: The key to stop pressing.
        """
        if key in self.long_press_flags:
            self.long_press_flags[key] = False  # Set the flag to False to stop pressing the key
            self.threads[key].join()  # Wait for the thread to finish

    def _press_key(self, key_char):
        """Auto-press a specific key based on its flag status."""
        while self.auto_press_flags.get(key_char, False):
            self.keyboard.press(key_char)
            self.keyboard.release(key_char)
            time.sleep(0.1)  # Prevent too rapid key presses
    def start_soft_press(self, key_char,duty_cycle=0.5):
        """
        Start soft-pressing and releasing a specified key in a duty cycle.
        :param key_char:
        :param duty_cycle:
        :return:
        """
        if key_char in self.soft_press_flags:
            if self.soft_press_flags[key_char]:
                return
        self.soft_press_flags[key_char] = True
        self.threads[key_char] =threading.Thread(target=self._soft_press, args=(key_char,duty_cycle,), daemon=True)
        self.threads[key_char].start()


    def _soft_press(self, key_char, duty_cycle):
        """
        Soft-press and release a key in a duty cycle.
        :param key_char:
        :param duty_cycle:
        :return:
        """
        while self.soft_press_flags.get(key_char, False):
            self.keyboard.press(key_char)
            time.sleep(duty_cycle)
            self.keyboard.release(key_char)
            time.sleep(1 - duty_cycle)

    def stop_soft_press(self, key_char):
        """
        Stop soft-pressing a specified key.
        :param key_char:
        :return:
        """
        if key_char in self.soft_press_flags:
            self.soft_press_flags[key_char] = False
            del self.threads[key_char]

    def _start_auto_press(self, key_char):
        """Start auto-pressing a specified key."""
        if key_char in self.auto_press_flags:
            return  # Already auto-pressing this key
        self.auto_press_flags[key_char] = True
        thread = threading.Thread(target=self.press_key, args=(key_char,), daemon=True)
        thread.start()
        self.threads[key_char] = thread

    def _stop_auto_press(self, key_char):
        """Stop auto-pressing a specified key."""
        if key_char in self.auto_press_flags:
            self.auto_press_flags[key_char] = False
            del self.threads[key_char]
    def press_and_release(self, key_char,duration=0.1):
        """Press and release a key."""

        if key_char in self.press_flags:
            if self.press_flags[key_char]:
                return

        #set flags
        self.press_flags[key_char] = True

        self.threads[key_char] = threading.Thread(target=self._press_and_release, args=(key_char,duration,), daemon=True)
        self.threads[key_char].start()


    def _press_and_release(self, key_char, duration):
        """Press and release a key."""
        self.keyboard.press(key_char)
        time.sleep(duration)
        self.keyboard.release(key_char)
        #reset flags
        self.press_flags[key_char] = False
        exit(0)
    def on_press(self, key):
        try:
            # Example: Use 's' key to stop all auto-presses
            if key.char == 's':
                for key_char in list(self.auto_press_flags.keys()):
                    self.stop_auto_press(key_char)
        except AttributeError:
            pass
        if self.user_only_mode:  # If in user-only mode, ignore program key presses
            return True

    def on_release(self, key):
        # Example: Stop listener if the escape key is pressed
        if key == Key.esc:
            return False

    def run(self):
        """Run the listener to manage auto-presses."""
        with Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            listener.join()

    def block_program_presses(self):
        """Enable user-only mode, blocking program key presses."""
        self.user_only_mode = True

    def allow_program_presses(self):
        """Disable user-only mode, allowing program key presses."""
        self.user_only_mode = False

# Example usage
# if __name__ == "__main__":
#     presser = AutoKeyPresser()
#     presser.start_auto_press('a')  # Start auto-pressing 'a'
#
#
#     time.sleep(2)
#     presser.start_auto_press('b')  # Start auto-pressing 'b'
#     time.sleep(1)
#     presser.stop_auto_press('a')  # Stop auto-pressing 'a'
#     presser.stop_auto_press('b')  # Stop auto-pressing 'b'
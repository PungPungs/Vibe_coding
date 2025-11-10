import pynput
from pynput.mouse import Button, Controller
import keyboard

mouse = Controller()
clicking = False

def toggle_clicking():
    global clicking
    clicking = not clicking
    if clicking:
        print("Holding down left mouse button...")
        mouse.press(Button.left)
    else:
        print("Releasing left mouse button...")
        mouse.release(Button.left)

def exit_macro():
    print("Esc pressed. Exiting macro.")
    if clicking:
        mouse.release(Button.left)
    exit(0)

keyboard.on_press_key('ctrl', lambda _:toggle_clicking())
keyboard.on_press_key('esc', lambda _:exit_macro())

print("Macro ready. Press Ctrl to toggle left mouse button, Esc to exit.")
keyboard.wait()
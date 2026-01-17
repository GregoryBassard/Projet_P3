import time
import pyautogui
import pydirectinput
import logging
import win32gui
import os

logger = logging.getLogger(__name__)

def save_replay(window):
    region = win32gui.GetWindowPlacement(window)[4]
    logger.info(f"Window region: {region}")
    inputs =  ['enter', 'enter', 'w', 'left', 'left', 'down', 'enter', 'down', 'down', 'down', 'down', 'enter']
    for key in inputs:
        time.sleep(0.01)
        pydirectinput.press(key)

    time.sleep(0.1)
    loc = None
    for img in os.listdir('script/TMInterface/img/'):
        logger.info(f"Trying to locate image: {img}")
        loc = locate_save_button(img)
        if loc is not None:
            pyautogui.moveTo(x=loc.x, y=loc.y)
            time.sleep(0.01)
            pyautogui.click()
            pydirectinput.press('enter')
            logger.info("Replay saved")
            break
        else:
            time.sleep(0.01)
    if loc is None:
        logger.error("Failed to locate save button after multiple attempts.")
        pydirectinput.press('escape')

def locate_save_button(img):
    try:
        loc = pyautogui.locateOnScreen(image=f'script/TMInterface/img/{img}', confidence=0.75)
        if loc:
            loc = pyautogui.center(loc)
            logger.info(f"Save button found at: {loc}")
            return loc
        else:
            logger.warning("Save button not found on screen.")
            return None
    except Exception as e:
        logger.error(f"Error locating save button: {e}")
        return None
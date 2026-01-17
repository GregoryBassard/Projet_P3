import time
import os
import subprocess
import logging
import win32gui

logger = logging.getLogger(__name__)

def run_TMInterface(tm_path):
    if os.path.exists(tm_path):
        cwd = os.path.dirname(tm_path)
        subprocess.Popen([tm_path], cwd=cwd)
        logger.info("TMInterface launched successfully")
    else:
        logger.info(f"TMInterface not found at: {tm_path}")

def check_TMInterface_running(max_attempts=20):
    for _ in range(max_attempts):
        time.sleep(1)
        window = win32gui.FindWindow(None, 'TmForever (TMInterface 1.4.3)')
        if window:
            logger.info("TMInterface window found!")
            return window
    return None

def start_TMInterface(tm_path=r'C:\Program Files (x86)\TmNationsForever\TMInterface.exe'):
    window = check_TMInterface_running()
    if window is not None:
        logger.info("TMInterface is already running.")
        return window
    

    logger.info("Starting TMInterface...")
    run_TMInterface(tm_path)
    
    window = check_TMInterface_running()
    
    if not window:
        logger.info("TMInterface window not found. Make sure TMInterface is running.")
        return None
    else:
        return window
    
def windows_resize(window, width=800, height=600):
    if window:
        win32gui.MoveWindow(window, 0, 0, width, height, True)
        logger.info(f"Resized window to {width}x{height}")
    else:
        logger.info("No window to resize.")

def windows_move(window, x=0, y=0):
    if window:
        rect = win32gui.GetWindowRect(window)
        width = rect[2] - rect[0]
        height = rect[3] - rect[1]
        win32gui.MoveWindow(window, x, y, width, height, True)
        logger.info(f"Moved window to ({x}, {y})")
    else:
        logger.info("No window to move.")

def get_screen_size():
    user32 = win32gui.GetDesktopWindow()
    rect = win32gui.GetWindowRect(user32)
    width = rect[2] - rect[0]
    height = rect[3] - rect[1]
    return width, height
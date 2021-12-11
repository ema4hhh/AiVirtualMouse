import pyautogui
import numpy as np
import cv2

while True:
  screenshot = pyautogui.screenshot(region=(1920, 1080, 1000, 1000))
  screenshot = np.array(screenshot)
  screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
  cv2.imshow("Screenshot", screenshot)
  if cv2.waitKey(1) & 0xFF == 27: 
    break

cv2.destroyAllWindows()
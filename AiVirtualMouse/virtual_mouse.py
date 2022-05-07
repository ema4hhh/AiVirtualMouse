import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time

prev_frame_time = 0
new_frame_time = 0

font = cv2.FONT_HERSHEY_SIMPLEX

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

color_mouse_pointer = (128, 0, 128)

#* Puntos de la pantalla-juego
SCREEN_GAME_X_INI = 0
SCREEN_GAME_Y_INI = 0
SCREEN_GAME_X_FIN = 1920
SCREEN_GAME_Y_FIN = 1080

aspect_ratio_screen = (SCREEN_GAME_X_FIN - SCREEN_GAME_X_INI) / (SCREEN_GAME_Y_FIN - SCREEN_GAME_Y_INI)

X_Y_INI = 20

def calculate_distance(x1, y1, x2, y2):
  p1 = np.array([x1, y1])
  p2 = np.array([x2, y2])
  return np.linalg.norm(p1 - p2)

def detect_finger_down(hand_landmarks):
  finger_down = False
  color_base = (255, 0, 112)
  color_index = (255, 198, 82)
  
  x_base1 = int(hand_landmarks.landmark[5].x * width)
  y_base1 = int(hand_landmarks.landmark[5].y * height)
  
  x_base2 = int(hand_landmarks.landmark[9].x * width)
  y_base2 = int(hand_landmarks.landmark[9].y * height)
  
  x_index1 = int(hand_landmarks.landmark[8].x * width)
  y_index1 = int(hand_landmarks.landmark[8].y * height)

  x_index2 = int(hand_landmarks.landmark[12].x * width)
  y_index2 = int(hand_landmarks.landmark[12].y * height)
  
  d_base = calculate_distance(x_base1, y_base1, x_base2, y_base2)
  d_base_index = calculate_distance(x_index1, y_index1, x_index2, y_index2)
  
  if d_base_index < d_base:
    finger_down = True
    color_base = (255, 0, 225)
    color_index = (255, 0, 225)
    
  cv2.circle(output, (x_base1, y_base1), 5, color_base, 2)
  cv2.circle(output, (x_index1, y_index1), 5, color_index, 2)
  cv2.line(output, (x_base1, y_base1), (x_base2, y_base2), color_base, 3)
  cv2.line(output, (x_base1, y_base1), (x_index1, y_index1), color_index, 3)
  
  return finger_down

with mp_hands.Hands(
  static_image_mode = False,
  model_complexity = 0,
  max_num_hands = 1,
  min_detection_confidence = 0.1) as hands:
  
  while True:
    ret, frame = cap.read()
    if ret == False:
      print("No camera detected")
      continue
    
    #Detects frames per second
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps = str(fps)

    height, width, _ = frame.shape
    frame = cv2.flip(frame, 1)
    
    # Drawing an area proportional to the area of the set
    area_width = width - X_Y_INI * 2
    area_height = int(area_width / aspect_ratio_screen)
    aux_image = np.zeros(frame.shape, np.uint8)
    aux_image = cv2.rectangle(aux_image, (X_Y_INI, X_Y_INI), (X_Y_INI + area_width, X_Y_INI + area_height), (255, 0, 0), -1)
    output =  cv2.addWeighted(frame, 1, aux_image, 0.7, 0)
    
    frame.flags.writeable = False
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame)

    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    if results.multi_hand_landmarks is not None:
      for hand_landmarks in results.multi_hand_landmarks:
        x = int(hand_landmarks.landmark[9].x * width)
        y = int(hand_landmarks.landmark[9].y * height)
        xm = np.interp(x, (X_Y_INI, X_Y_INI + area_width), (SCREEN_GAME_X_INI, SCREEN_GAME_X_FIN))
        ym = np.interp(y, (X_Y_INI, X_Y_INI + area_height), (SCREEN_GAME_Y_INI, SCREEN_GAME_Y_FIN))
        pyautogui.moveTo(int(xm), int(ym))
        if detect_finger_down(hand_landmarks):
          pyautogui.click()
        
        cv2.circle(output, (x, y), 10, color_mouse_pointer, 3)
        cv2.circle(output, (x, y), 5, color_mouse_pointer, -1)
    
    #cv2.imshow("Frame", frame)
    cv2.putText(output, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
    cv2.imshow("Output", output)
    
    if cv2.waitKey(33) == ord("q"):
      break
    
cap.release()
cv2.destroyAllWindows()
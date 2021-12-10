import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
  static_image_mode = False,
  max_num_hands = 2,
  min_detection_confidence = 0.5) as hands:
  
  while True:
    
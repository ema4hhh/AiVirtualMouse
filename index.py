import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

with mp_hands.Hands(
  static_image_mode = True,
  max_num_hands = 2,
  min_detection_confidence = 0.5) as hands:
  
  image = cv2.imread('/home/emanuel/Escritorio/IaVirtualMouse/hands.jpg')
  height, width, _ = image.shape
  
  image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  
  results = hands.process(image_rgb)
  
  #Print handedness
  #print("Handness: ", results.multi_handedness)
  #Prin landmark's position
  #print("Landmarks:", results.multi_hand_landmarks)
  
  if results.multi_hand_landmarks is not None:
    for hand_landmarks in results.multi_hand_landmarks:
      mp_drawing.draw_landmarks(
        image, hand_landmarks, mp_hands.HAND_CONNECTIONS
      )
  
  image = cv2.flip(image, 1)
  
  
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
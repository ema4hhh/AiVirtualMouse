import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

with mp_hands.Hands(
  static_image_mode = True,
  max_num_hands = 2,
  min_detection_confidence = 0.5) as hands:
  
  image = cv2.imread('/home/emanuel/Escritorio/IaVirtualMouse/Learning_to_use_mediapipe/hands.jpg')
  height, width, _ = image.shape
  
  image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  
  results = hands.process(image_rgb)
  
  #Print handedness
  print("Handness: ", results.multi_handedness)
  
  #Prin landmark's position
  #print("Landmarks:", results.multi_hand_landmarks)
  
  if results.multi_hand_landmarks is not None:
    for hand_landmarks in results.multi_hand_landmarks:
      #! Change connections color
      """ 
      mp_drawing.draw_landmarks(
        image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(128,117,128), thickness=2, circle_radius=3),
        mp_drawing.DrawingSpec(color=(128, 0, 128), thickness=2)
      ) 
      """
      #----------------------------------------------------------------------------
    
      #! Geting access to key points with their names
      """
      x1 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * width)
      y1 = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * height)
      
      x2 = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width)
      y2 = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height)
      
      x3 = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * width)
      y3 = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * height)
      
      x4 = int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * width)
      y4 = int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * height)
      
      x5 = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * width)
      y5 = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * height)
      
      cv2.circle(image, (x1, y1), 3, (128, 0, 128), 3)
      cv2.circle(image, (x2, y2), 3, (128, 0, 128), 3)
      cv2.circle(image, (x3, y3), 3, (128, 0, 128), 3)
      cv2.circle(image, (x4, y4), 3, (128, 0, 128), 3)
      cv2.circle(image, (x5, y5), 3, (128, 0, 128), 3)
      """
      
      #! Getting access to key points with their index
      index = [4, 8, 12, 16, 20] #These are the TIP index, see mediapipeNames.jpg
      for (i, points) in enumerate(hand_landmarks.landmark):
        if i in index:
          x = int(points.x * width)
          y = int(points.y * height)
          cv2.circle(image, (x,y), 3, (128, 0, 128), 3)
        
      
  else:
    print("The image doesn't contains a hand to recognize")
      
  
  image = cv2.flip(image, 1)
  
  
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
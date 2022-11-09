import cv2
import numpy as np
import mediapipe as mp

from utils import *
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

class point:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
    
    def __str__(self) -> str:
   		return f"({self.x}, {self.y})"

class Config:
    img_height = 480
    img_width = 640
    img_channel = 3
    img_shape = (img_height, img_width, img_channel)
    img_size = 480 * 640 * 3

    box_margin = 150
    box_alpha = 0.7
    model_config = {
    'model_complexity': 0,
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5
    }

def get_landmark_array(landmarks, image_shape, normal_x, normal_y):

    image_width, image_height = image_shape[1], image_shape[0]
    landmark_list = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1) + normal_x
        landmark_y = min(int(landmark.y * image_height), image_height - 1) + normal_y
        landmark_list.append((landmark_x, landmark_y))

    return np.array(landmark_list)

def calc_bounding_rect(landmark_array, p1, p2):

    x, y, w, h = cv2.boundingRect(landmark_array)
    p1.x = int(Config.box_alpha * p1.x + (1 - Config.box_alpha) * x)
    p1.y = int(Config.box_alpha * p1.y + (1 - Config.box_alpha) * y)
    p2.x = int(Config.box_alpha * p2.x + (1 - Config.box_alpha) * (x + w))
    p2.y = int(Config.box_alpha * p2.y + (1 - Config.box_alpha) * (y + h))

    return p1, p2

p1, p2 = point(0, 0), point(Config.img_width - 1, Config.img_height - 1)

cap = cv2.VideoCapture(0)
with mp_hands.Hands(**Config.model_config) as hands:
    while cap.isOpened():
        success, image = cap.read()
        normal_x, normal_y = max(p1.x - Config.box_margin, 0), max(p1.y - Config.box_margin, 0)
        cropped_image = image[normal_y:min(p2.y + Config.box_margin, Config.img_height), normal_x:min(p2.x + Config.box_margin, Config.img_width), ::]

        results = hands.process(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmark_array = get_landmark_array(hand_landmarks, cropped_image.shape, normal_x, normal_y)
                p1, p2 = calc_bounding_rect(landmark_array, p1, p2)
                cv2.rectangle(image, (normal_x, normal_y), (min(p2.x + Config.box_margin, Config.img_width), min(p2.y + Config.box_margin, Config.img_height)), (0, 0, 0), 1)
                draw_landmarks(image, landmark_array)

        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(1) & 0xFF == 27:
            break
cap.release()

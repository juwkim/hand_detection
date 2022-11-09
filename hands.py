import cv2
import numpy as np
from mediapipe.python.solutions import hands as mp_hands

class point:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
    
    def __str__(self) -> str:
   		return f"({self.x}, {self.y})"

class Config:
    img_height = 720
    img_width = 1280
    img_channel = 3
    img_shape = (img_height, img_width, img_channel)

    box_margin = 50
    box_alpha = 0.7
    model_config = {
    'model_complexity': 1,
    'min_detection_confidence': 0.3,
    'min_tracking_confidence': 0.3
    }

def get_landmark_array(landmarks, image_shape, normal_x, normal_y):

    image_width, image_height = image_shape[1], image_shape[0]
    landmark_list = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = int(landmark.x * image_width) + normal_x
        landmark_y = int(landmark.y * image_height) + normal_y
        landmark_list.append((landmark_x, landmark_y))
    return np.array(landmark_list)

def get_bounding_rect(landmark_array, p1, p2):

    x, y, w, h = cv2.boundingRect(landmark_array)
    p1.x = int(Config.box_alpha * p1.x + (1 - Config.box_alpha) * x)
    p1.y = int(Config.box_alpha * p1.y + (1 - Config.box_alpha) * y)
    p2.x = int(Config.box_alpha * p2.x + (1 - Config.box_alpha) * (x + w))
    p2.y = int(Config.box_alpha * p2.y + (1 - Config.box_alpha) * (y + h))

    return p1, p2

def draw_landmarks(image, landmark_array, line_thickness = 1, vertex_radius = 2, fingertip_point_radius = 3):

    # Thumb
    cv2.line(image, landmark_array[2], landmark_array[3], (0, 0, 0), line_thickness * 2)
    cv2.line(image, landmark_array[2], landmark_array[3], (255, 255, 255), line_thickness)
    cv2.line(image, landmark_array[3], landmark_array[4], (0, 0, 0), line_thickness * 2)
    cv2.line(image, landmark_array[3], landmark_array[4], (255, 255, 255), line_thickness)

    # Index finger
    cv2.line(image, landmark_array[5], landmark_array[6], (0, 0, 0), line_thickness * 2)
    cv2.line(image, landmark_array[5], landmark_array[6], (255, 255, 255), line_thickness)
    cv2.line(image, landmark_array[6], landmark_array[7], (0, 0, 0), line_thickness * 2)
    cv2.line(image, landmark_array[6], landmark_array[7], (255, 255, 255), line_thickness)
    cv2.line(image, landmark_array[7], landmark_array[8], (0, 0, 0), line_thickness * 2)
    cv2.line(image, landmark_array[7], landmark_array[8], (255, 255, 255), line_thickness)

    # Middle finger
    cv2.line(image, landmark_array[9], landmark_array[10], (0, 0, 0), line_thickness * 2)
    cv2.line(image, landmark_array[9], landmark_array[10], (255, 255, 255), line_thickness)
    cv2.line(image, landmark_array[10], landmark_array[11], (0, 0, 0), line_thickness * 2)
    cv2.line(image, landmark_array[10], landmark_array[11], (255, 255, 255), line_thickness)
    cv2.line(image, landmark_array[11], landmark_array[12], (0, 0, 0), line_thickness * 2)
    cv2.line(image, landmark_array[11], landmark_array[12], (255, 255, 255), line_thickness)

    # Ring finger
    cv2.line(image, landmark_array[13], landmark_array[14], (0, 0, 0), line_thickness * 2)
    cv2.line(image, landmark_array[13], landmark_array[14], (255, 255, 255), line_thickness)
    cv2.line(image, landmark_array[14], landmark_array[15], (0, 0, 0), line_thickness * 2)
    cv2.line(image, landmark_array[14], landmark_array[15], (255, 255, 255), line_thickness)
    cv2.line(image, landmark_array[15], landmark_array[16], (0, 0, 0), line_thickness * 2)
    cv2.line(image, landmark_array[15], landmark_array[16], (255, 255, 255), line_thickness)

    # Little finger
    cv2.line(image, landmark_array[17], landmark_array[18], (0, 0, 0), line_thickness * 2)
    cv2.line(image, landmark_array[17], landmark_array[18], (255, 255, 255), line_thickness)
    cv2.line(image, landmark_array[18], landmark_array[19], (0, 0, 0), line_thickness * 2)
    cv2.line(image, landmark_array[18], landmark_array[19], (255, 255, 255), line_thickness)
    cv2.line(image, landmark_array[19], landmark_array[20], (0, 0, 0), line_thickness * 2)
    cv2.line(image, landmark_array[19], landmark_array[20], (255, 255, 255), line_thickness)

    # Palm
    cv2.line(image, landmark_array[0], landmark_array[1], (0, 0, 0), line_thickness * 2)
    cv2.line(image, landmark_array[0], landmark_array[1], (255, 255, 255), line_thickness)
    cv2.line(image, landmark_array[1], landmark_array[2], (0, 0, 0), line_thickness * 2)
    cv2.line(image, landmark_array[1], landmark_array[2], (255, 255, 255), line_thickness)
    cv2.line(image, landmark_array[2], landmark_array[5], (0, 0, 0), line_thickness * 2)
    cv2.line(image, landmark_array[2], landmark_array[5], (255, 255, 255), line_thickness)
    cv2.line(image, landmark_array[5], landmark_array[9], (0, 0, 0), line_thickness * 2)
    cv2.line(image, landmark_array[5], landmark_array[9], (255, 255, 255), line_thickness)
    cv2.line(image, landmark_array[9], landmark_array[13], (0, 0, 0), line_thickness * 2)
    cv2.line(image, landmark_array[9], landmark_array[13], (255, 255, 255), line_thickness)
    cv2.line(image, landmark_array[13], landmark_array[17], (0, 0, 0), line_thickness * 2)
    cv2.line(image, landmark_array[13], landmark_array[17], (255, 255, 255), line_thickness)
    cv2.line(image, landmark_array[17], landmark_array[0], (0, 0, 0), line_thickness * 2)
    cv2.line(image, landmark_array[17], landmark_array[0], (255, 255, 255), line_thickness)


    cv2.circle(image, landmark_array[0], vertex_radius, (255, 255, 255), -1)
    cv2.circle(image, landmark_array[0], vertex_radius, (0, 0, 0), 1)

    cv2.circle(image, landmark_array[1], vertex_radius, (255, 255, 255), -1)
    cv2.circle(image, landmark_array[1], vertex_radius, (0, 0, 0), 1)

    cv2.circle(image, landmark_array[2], vertex_radius, (255, 255, 255), -1)
    cv2.circle(image, landmark_array[2], vertex_radius, (0, 0, 0), 1)

    cv2.circle(image, landmark_array[3], vertex_radius, (255, 255, 255), -1)
    cv2.circle(image, landmark_array[3], vertex_radius, (0, 0, 0), 1)

    cv2.circle(image, landmark_array[4], fingertip_point_radius, (255, 255, 255), -1)
    cv2.circle(image, landmark_array[4], fingertip_point_radius, (0, 0, 0), 1)

    cv2.circle(image, landmark_array[5], vertex_radius, (255, 255, 255), -1)
    cv2.circle(image, landmark_array[5], vertex_radius, (0, 0, 0), 1)

    cv2.circle(image, landmark_array[6], vertex_radius, (255, 255, 255), -1)
    cv2.circle(image, landmark_array[6], vertex_radius, (0, 0, 0), 1)

    cv2.circle(image, landmark_array[7], vertex_radius, (255, 255, 255), -1)
    cv2.circle(image, landmark_array[7], vertex_radius, (0, 0, 0), 1)

    cv2.circle(image, landmark_array[8], fingertip_point_radius, (255, 255, 255), -1)
    cv2.circle(image, landmark_array[8], fingertip_point_radius, (0, 0, 0), 1)

    cv2.circle(image, landmark_array[9], vertex_radius, (255, 255, 255), -1)
    cv2.circle(image, landmark_array[9], vertex_radius, (0, 0, 0), 1)

    cv2.circle(image, landmark_array[10], vertex_radius, (255, 255, 255), -1)
    cv2.circle(image, landmark_array[10], vertex_radius, (0, 0, 0), 1)

    cv2.circle(image, landmark_array[11], vertex_radius, (255, 255, 255), -1)
    cv2.circle(image, landmark_array[11], vertex_radius, (0, 0, 0), 1)

    cv2.circle(image, landmark_array[12], fingertip_point_radius, (255, 255, 255), -1)
    cv2.circle(image, landmark_array[12], fingertip_point_radius, (0, 0, 0), 1)

    cv2.circle(image, landmark_array[13], vertex_radius, (255, 255, 255), -1)
    cv2.circle(image, landmark_array[13], vertex_radius, (0, 0, 0), 1)

    cv2.circle(image, landmark_array[14], vertex_radius, (255, 255, 255), -1)
    cv2.circle(image, landmark_array[14], vertex_radius, (0, 0, 0), 1)

    cv2.circle(image, landmark_array[15], vertex_radius, (255, 255, 255), -1)
    cv2.circle(image, landmark_array[15], vertex_radius, (0, 0, 0), 1)

    cv2.circle(image, landmark_array[16], fingertip_point_radius, (255, 255, 255), -1)
    cv2.circle(image, landmark_array[16], fingertip_point_radius, (0, 0, 0), 1)

    cv2.circle(image, landmark_array[17], vertex_radius, (255, 255, 255), -1)
    cv2.circle(image, landmark_array[17], vertex_radius, (0, 0, 0), 1)

    cv2.circle(image, landmark_array[18], vertex_radius, (255, 255, 255), -1)
    cv2.circle(image, landmark_array[18], vertex_radius, (0, 0, 0), 1)

    cv2.circle(image, landmark_array[19], vertex_radius, (255, 255, 255), -1)
    cv2.circle(image, landmark_array[19], vertex_radius, (0, 0, 0), 1)

    cv2.circle(image, landmark_array[20], fingertip_point_radius, (255, 255, 255), -1)
    cv2.circle(image, landmark_array[20], fingertip_point_radius, (0, 0, 0), 1)

def draw_rectangle(image, p1, p2):
    cv2.rectangle(image, (p1.x - Config.box_margin, p1.y - Config.box_margin), (p2.x + Config.box_margin, p2.y + Config.box_margin), color=(0, 0, 0), thickness=1)

def detect_hands():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)

    p1, p2 = point(0, 0), point(Config.img_width - 1, Config.img_height - 1)
    with mp_hands.Hands(**Config.model_config) as hands:
        while cap.isOpened():
            success, image = cap.read()

            normal_x, normal_y = max(p1.x - Config.box_margin, 0), max(p1.y - Config.box_margin, 0)
            cropped_image = image[normal_y:p2.y + Config.box_margin, normal_x:p2.x + Config.box_margin, ::]

            results = hands.process(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmark_array = get_landmark_array(hand_landmarks, cropped_image.shape, normal_x, normal_y)
                    p1, p2 = get_bounding_rect(landmark_array, p1, p2)
                    draw_rectangle(image, p1, p2)
                    draw_landmarks(image, landmark_array)
            else:
                p1, p2 = point(0, 0), point(Config.img_width - 1, Config.img_height - 1)

            cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
            if cv2.waitKey(1) & 0xFF == 27:
                break
    cap.release()

if __name__ == "__main__":
    detect_hands()
import cv2
import mediapipe as mp


class HandTracker:

    def __init__(self,
                 max_num_hands=1,
                 detection_confidence=0.7,
                 tracking_confidence=0.7):

        self.cap = cv2.VideoCapture(0)

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )

    def get_frame(self):
      
        success, frame = self.cap.read()

        if not success:
            return None

        frame = cv2.flip(frame, 1)

        return frame

    def detect(self, frame):
       

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.hands.process(rgb_frame)

        return results

    def get_index_tip(self, frame, results):
     

        if not results.multi_hand_landmarks:
            return None

        hand = results.multi_hand_landmarks[0]

        self.mp_draw.draw_landmarks(
            frame,
            hand,
            self.mp_hands.HAND_CONNECTIONS
        )

        height, width, _ = frame.shape

        fingertip = hand.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]

        x = int(fingertip.x * width)
        y = int(fingertip.y * height)

        cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)

        return (x, y)

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
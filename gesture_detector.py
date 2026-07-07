from mediapipe.python.solutions.hands import HandLandmark


class GestureDetector:

    @staticmethod
    def finger_up(hand, tip, pip):
        return hand.landmark[tip].y < hand.landmark[pip].y

    def detect(self, hand):

        index = self.finger_up(
            hand,
            HandLandmark.INDEX_FINGER_TIP,
            HandLandmark.INDEX_FINGER_PIP
        )

        middle = self.finger_up(
            hand,
            HandLandmark.MIDDLE_FINGER_TIP,
            HandLandmark.MIDDLE_FINGER_PIP
        )

        ring = self.finger_up(
            hand,
            HandLandmark.RING_FINGER_TIP,
            HandLandmark.RING_FINGER_PIP
        )

        pinky = self.finger_up(
            hand,
            HandLandmark.PINKY_TIP,
            HandLandmark.PINKY_PIP
        )

        if index and not middle and not ring and not pinky:
            return "DRAW"

        if index and middle and not ring and not pinky:
            return "MOVE"

        if not index and not middle and not ring and not pinky:
            return "CLEAR"

        return "NONE"
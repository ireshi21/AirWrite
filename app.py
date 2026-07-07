import cv2

from tracker import HandTracker
from canvas import DrawingCanvas
from gesture_detector import GestureDetector
from segmenter import CharacterSegmenter
from recognizer import CharacterRecognizer


print("Starting AirWrite...")

tracker = HandTracker()
canvas = DrawingCanvas()
detector = GestureDetector()
segmenter = CharacterSegmenter()
recognizer = CharacterRecognizer()


while True:
    frame = tracker.get_frame()

    if frame is None:
        print("Could not read camera frame.")
        break

    results = tracker.detect(frame)

    hand = tracker.get_hand(results)
    point = tracker.get_index_tip(frame, results)

    gesture = detector.detect(hand) if hand else "NONE"

    if gesture == "DRAW":
        canvas.draw(point)

    elif gesture == "MOVE":
        canvas.draw(None)

    elif gesture == "CLEAR":
        canvas.clear()

    else:
        canvas.draw(None)

    frame = canvas.overlay(frame)

    cv2.putText(
        frame,
        f"Gesture: {gesture}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Press S to recognize | ESC to quit",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    cv2.imshow("AirWrite", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        character_image = segmenter.extract(canvas.get_image())

        if character_image is None:
            print("Nothing to recognize.")

        else:
            predicted_character, confidence = recognizer.predict(
                character_image
            )

            print(
                f"Prediction: {predicted_character} | "
                f"Confidence: {confidence:.2%}"
            )

            preview = cv2.resize(
                character_image,
                (280, 280),
                interpolation=cv2.INTER_NEAREST
            )

            cv2.imshow("CNN Input", preview)

    elif key == 27:
        break


tracker.release()
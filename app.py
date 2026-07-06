
import cv2


from tracker import HandTracker


tracker = HandTracker()


frame = tracker.get_frame()



import cv2
from tracker import HandTracker

print("Starting AirWrite...")

tracker = HandTracker()

while True:
    frame = tracker.get_frame()

    if frame is None:
        print("No frame received.")
        break

    results = tracker.detect(frame)

    point = tracker.get_index_tip(frame, results)

    if point:
        cv2.putText(
            frame,
            f"Index: {point}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("AirWrite", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

tracker.release()
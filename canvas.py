import cv2
import numpy as np


class DrawingCanvas:

    def __init__(self):
        self.canvas = None
        self.previous_point = None

    def draw(self, point):

        if self.canvas is None:
            return

        if point is None:
            self.previous_point = None
            return

        if self.previous_point is None:
            self.previous_point = point
            return

        cv2.line(
            self.canvas,
            self.previous_point,
            point,
            (0, 255, 255),
            5
        )

        self.previous_point = point
    def get_image(self):
        if self.canvas is None:
            return None

        return self.canvas.copy()
    def clear(self):

        if self.canvas is not None:
            self.canvas[:] = 0

        self.previous_point = None

    def overlay(self, frame):

        if self.canvas is None:
            self.canvas = np.zeros_like(frame)

        return cv2.addWeighted(
            frame,
            1,
            self.canvas,
            1,
            0
        )
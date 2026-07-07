import cv2
import numpy as np


class CharacterSegmenter:

    def extract(self, canvas):
        if canvas is None:
            return None

        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(
            gray,
            20,
            255,
            cv2.THRESH_BINARY
        )

        points = cv2.findNonZero(binary)

        if points is None:
            return None

        x, y, width, height = cv2.boundingRect(points)
        character = binary[y:y + height, x:x + width]

        # Fit the longest side inside a 20x20 region.
        target_size = 20

        if width > height:
            new_width = target_size
            new_height = max(1, int(height * target_size / width))
        else:
            new_height = target_size
            new_width = max(1, int(width * target_size / height))

        character = cv2.resize(
            character,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

        # Center inside a 28x28 image.
        output = np.zeros((28, 28), dtype=np.uint8)

        x_offset = (28 - new_width) // 2
        y_offset = (28 - new_height) // 2

        output[
            y_offset:y_offset + new_height,
            x_offset:x_offset + new_width
        ] = character

        return output
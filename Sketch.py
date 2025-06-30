import cv2
import numpy as np
from datetime import datetime


def pencil_sketch(frame):
    """
    Convert an image to a pencil sketch using adaptive thresholding.
    """
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray_img, 7)
    edges = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 7
    )
    return edges


def canny_sketch(frame):
    """
    Convert an image to a sketch using Canny edge detection.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges


def dodge_blend_sketch(frame):
    """
    Convert an image to a sketch using color dodge blending.
    """
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    invert = cv2.bitwise_not(gray_img)
    blur = cv2.GaussianBlur(invert, (21, 21), 0)
    inverted_blur = cv2.bitwise_not(blur)
    sketch = cv2.divide(gray_img, inverted_blur, scale=256.0)
    return sketch


def main():
    # Try to open the camera and handle errors
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open camera.")
        return

    effect_mode = 0  # 0: pencil_sketch, 1: canny_sketch, 2: dodge_blend_sketch
    effect_names = ["Pencil Sketch", "Canny Sketch", "Dodge Blend Sketch"]

    while True:
        ret, img = camera.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Switch between sketch effects
        if effect_mode == 0:
            sketch_img = pencil_sketch(img)
        elif effect_mode == 1:
            sketch_img = canny_sketch(img)
        else:
            sketch_img = dodge_blend_sketch(img)

        combined = np.hstack((cv2.cvtColor(sketch_img, cv2.COLOR_GRAY2BGR), img))
        cv2.imshow(f"Original | {effect_names[effect_mode]}", combined)
        key = cv2.waitKey(1)
        if key == 27:  # ESC to exit
            break
        elif key == ord("s"):
            filename = f"sketch_{effect_names[effect_mode].replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(filename, sketch_img)
            print(f"Saved {filename}")
        elif key == ord("e"):
            effect_mode = (effect_mode + 1) % 3  # Cycle through effects
            print(f"Switched to: {effect_names[effect_mode]}")

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
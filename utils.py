import cv2
import numpy as np
from PIL import Image
from io import BytesIO

# Define the class name to integer mapping
class_name_to_int = {
    "R": 0,
    "C": 1,
    "IC": 2,
    "Q": 3,
    "J": 4,
    "L": 5,
    "RN": 6,
    "D": 7,
    "TP": 8,
    "CR": 9,
    "BTN": 10,
    "T": 11,
    "F": 12,
    "P": 13,
    "LED": 14,
}

color_values = {
    0: (255, 0, 0),  # R
    1: (255, 255, 0),  # C
    2: (185, 215, 237),  # IC
    3: (170, 0, 255),  # Q
    4: (255, 127, 0),  # J
    5: (191, 255, 0),  # L
    6: (0, 64, 255),  # RN
    7: (106, 255, 0),  # D
    8: (237, 185, 185),  # TP
    9: (220, 185, 237),  # CR
    10: (143, 35, 35),  # BTN
    11: (79, 143, 35),  # T
    12: (115, 115, 115),  # F
    13: (231, 233, 185),  # P
    14: (245, 130, 48),  # LED
}

# Reverse mapping for convenience
int_to_class_name = {v: k for k, v in class_name_to_int.items()}


def map_classes_to_int(classes_list):
    return [class_name_to_int[cls] for cls in classes_list if cls in class_name_to_int]


def read_imagefile(file) -> np.ndarray:
    """
    Read uploaded image file and convert it to OpenCV format.

    Args:
        file (UploadFile): Uploaded image file.

    Returns:
        np.ndarray: Image in OpenCV format (BGR).
    """
    image = Image.open(BytesIO(file))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def resize_image(image: np.ndarray, max_edge_length: int) -> np.ndarray:
    """
    Resize the image so that the longer edge is equal to max_edge_length,
    maintaining the aspect ratio.
    """
    h, w = image.shape[:2]
    if h > w:
        scale = max_edge_length / h
        new_size = (int(w * scale), max_edge_length)
    else:
        scale = max_edge_length / w
        new_size = (max_edge_length, int(h * scale))

    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    return resized_image

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import math


def cv2pil(image):
    # Convert Opencv image to PIL image

    new_image = image.copy()
    if new_image.ndim == 2:
        pass

    elif new_image.shape[2] == 3:  # Color
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # Transparent
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image

def pil2cv(image):
    ''' Convert PIL image to OpenCV image '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # Grayscale
        pass
    elif new_image.shape[2] == 3:  # Color
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # Transparent
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image


def expand_box(head_box, width, height, expand_size):
    ''' Expand bounding box by a fixed size '''
    x0, y0, x1, y1 = head_box
    x0 = max(x0 - expand_size, 0)
    y0 = max(y0 - expand_size, 0)
    x1 = min(x1 + expand_size, width)
    y1 = min(y1 + expand_size, height)
    return [x0, y0, x1, y1]



def expand_box_ratio(head_box, width, height, expand_ratio):
    ''' Expand bounding box by a given ratio '''
    x0, y0, x1, y1 = head_box
    box_w = x1 - x0
    box_h = y1 - y0
    expand_size_w = int(box_w * expand_ratio)
    expand_size_h = int(box_h * expand_ratio)
    x0 = max(x0 - expand_size_w, 0)
    y0 = max(y0 - expand_size_h, 0)
    x1 = min(x1 + expand_size_w, width)
    y1 = min(y1 + expand_size_h, height)
    return [x0, y0, x1, y1]

def get_angle(x0, y0, x1, y1, x2, y2):
    ''' Calculate the angle between two vectors originating from a common point '''
    vec1 = [x1 - x0, y1 - y0]
    vec2 = [x2 - x0, y2 - y0]
    absvec1 = np.linalg.norm(vec1)
    absvec2 = np.linalg.norm(vec2)
    if absvec1 == 0 or absvec2 == 0:
        raise ValueError("Vector length cannot be zero")
    inner = np.inner(vec1, vec2)
    cos_theta = inner / (absvec1 * absvec2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Ensure cos_theta is within the valid range
    theta = math.degrees(math.acos(cos_theta))
    return theta
    
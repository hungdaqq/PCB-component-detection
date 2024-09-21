from fastapi import (
    FastAPI,
    APIRouter,
    File,
    UploadFile,
    HTTPException,
    Response,
    Form,
)
import uvicorn
from typing import Optional
from collections import Counter
import cv2
import numpy as np
from ultralytics import YOLO
from io import BytesIO
from PIL import Image
import easyocr
import torch

# Initialize FastAPI app
router = APIRouter(prefix="/api/v1")
# Load the YOLO model
pcb_segmentation_model = YOLO("./weights/pcb_segmentation_best.pt")
pcb_detection_model = YOLO("./weights/pcb_detection_best.pt")


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


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        image = read_imagefile(await file.read())

        # Predict on the image
        results = pcb_segmentation_model.predict(source=image)
        # Extract prediction data
        predictions = []
        for box, mask in zip(results[0].boxes, results[0].masks):
            predictions.append(
                {
                    "class_id": int(box.cls[0]),
                    "class_name": int_to_class_name[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bounding_box": {
                        "xmin": int(box.xyxy[0][0]),
                        "ymin": int(box.xyxy[0][1]),
                        "xmax": int(box.xyxy[0][2]),
                        "ymax": int(box.xyxy[0][3]),
                    },
                    "mask": mask.xy.pop().reshape(-1, 1, 2).tolist(),
                }
            )

        # Count occurrences
        counter = Counter(results[0].boxes.cls.tolist())

        # Initialize the result dictionary with all class names set to 0
        result = {class_name: 0 for class_name in class_name_to_int}

        # Update counts based on the actual occurrences in the list
        for num, count in counter.items():
            if num in int_to_class_name:
                result[int_to_class_name[int(num)]] = count

        return {
            "image_shape": {
                "height": results[0].orig_shape[0],
                "width": results[0].orig_shape[1],
            },
            "speed": results[0].speed,
            "appearances": result,
            "predictions": predictions,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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

# Reverse mapping for convenience
int_to_class_name = {v: k for k, v in class_name_to_int.items()}


def map_classes_to_int(classes_list):
    return [class_name_to_int[cls] for cls in classes_list if cls in class_name_to_int]


@router.post("/predict-detection")
async def predict_png(
    file: UploadFile = File(...),
    img_size: Optional[int] = Form(1280),
    conf: Optional[float] = Form(0.25),
    iou: Optional[float] = Form(0.45),
    show_conf: Optional[bool] = Form(True),
    show_labels: Optional[bool] = Form(True),
    show_boxes: Optional[bool] = Form(True),
    line_width: Optional[int] = Form(None),
    classes: Optional[str] = Form([]),
):
    try:
        if line_width == None:
            line_width = img_size // 60

        # Read the uploaded file
        image = read_imagefile(await file.read())

        # Predict on the image
        if classes != []:
            results = pcb_detection_model.predict(
                source=image,
                conf=conf,
                iou=iou,
                classes=map_classes_to_int(classes),
            )
        else:
            results = pcb_detection_model.predict(
                source=image,
                conf=conf,
                iou=iou,
            )
        # Extract prediction data
        img_with_boxes = results[0].plot(
            boxes=show_boxes,
            labels=show_labels,
            conf=show_conf,
            line_width=line_width,
        )

        # Resize image
        resized_image = resize_image(img_with_boxes, img_size)

        _, img_encoded = cv2.imencode(".png", resized_image)
        img_bytes = img_encoded.tobytes()

        return Response(content=img_bytes, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Function to assign colors to segmentation masks
def assign_colors_to_masks(masks, classes, color_values):
    # Create a blank canvas for the colored mask
    height, width = masks.shape[1], masks.shape[2]
    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)

    # Loop through each mask and assign corresponding color based on class label
    for idx, mask in enumerate(masks):
        class_idx = int(classes[idx])  # Get the class index of the mask
        color = color_values.get(
            class_idx, (0, 0, 0)
        )  # Default to black if class not in color_values
        colored_mask[mask == 1] = color  # Apply color where mask is 1 (object present)

    return colored_mask


# Define the color mapping (example with 26 classes)
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


@router.post("/predict-segmentation")
async def predict_segmentation(
    file: UploadFile = File(...),
    img_size: Optional[int] = Form(1280),
    show_detection: Optional[bool] = Form(False),
    show_boxes: Optional[bool] = Form(False),
    show_conf: Optional[bool] = Form(True),
    show_labels: Optional[bool] = Form(True),
    show_masks: Optional[bool] = Form(True),
    show_segmentation: Optional[bool] = Form(False),
    line_width: Optional[int] = Form(None),
    classes: Optional[str] = Form([]),
):
    try:
        # Read the uploaded file
        image = read_imagefile(await file.read())

        if classes != []:
            results = pcb_segmentation_model.predict(
                source=image,
                classes=map_classes_to_int(classes),
            )
        else:
            results = pcb_segmentation_model.predict(source=image)

        # Resize image to match mask dimensions if needed

        if show_segmentation:
            masks = results[0].masks.data.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            colored_mask = assign_colors_to_masks(masks, classes, color_values)
            image_rgb = cv2.resize(
                image, (colored_mask.shape[1], colored_mask.shape[0])
            )
            result_image = cv2.addWeighted(image_rgb, 0.2, colored_mask, 0.8, 0)

        if show_detection:
            result_image = results[0].plot(
                boxes=show_boxes,
                masks=show_masks,
                labels=show_labels,
                conf=show_conf,
                line_width=line_width,
            )

        resized_image = resize_image(result_image, img_size)
        _, img_encoded = cv2.imencode(".png", resized_image)
        img_bytes = img_encoded.tobytes()

        return Response(content=img_bytes, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


app = FastAPI()
app.include_router(router)

if __name__ == "__main__":
    # Run the FastAPI app with Uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

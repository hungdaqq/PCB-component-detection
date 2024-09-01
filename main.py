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

# Initialize the reader
reader = easyocr.Reader(['en']) 

# Initialize FastAPI app
router = APIRouter(prefix="/api/v1")

# Load the YOLO model
model = YOLO("./weights/best.pt")


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
        results = model.predict(source=image, conf=0.3, iou=0.7)
        # Extract prediction data
        predictions = []
        for box in results[0].boxes:
            predictions.append(
                {
                    "class_id": int(box.cls[0]),
                    "confidence": float(box.conf[0]),
                    "bounding_box": {
                        "xmin": int(box.xyxy[0][0]),
                        "ymin": int(box.xyxy[0][1]),
                        "xmax": int(box.xyxy[0][2]),
                        "ymax": int(box.xyxy[0][3]),
                    },
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

        # Perform OCR on the image
        ocr_results = reader.readtext(image)

        # Format the OCR results
        ocr_predictions = []
        for bbox, text, prob in ocr_results:
            ocr_predictions.append({
                "text": text,
                "confidence": prob,
                "bounding_box": {
                    "xmin": int(bbox[0][0]),
                    "ymin": int(bbox[0][1]),
                    "xmax": int(bbox[2][0]),
                    "ymax": int(bbox[2][1]),
                },
            })

        return {
            "image_shape": {
                "height": results[0].orig_shape[0],
                "width": results[0].orig_shape[1],
            },
            "speed": results[0].speed,
            "appearances": result,
            "predictions": predictions,
            "ocr_results": ocr_predictions,  # Include OCR results in the response
        }


    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Define the class name to integer mapping
class_name_to_int = {
    "R": 0,
    "C": 1,
    "U": 2,
    "Q": 3,
    "J": 4,
    "L": 5,
    "RA": 6,
    "D": 7,
    "RN": 8,
    "TP": 9,
    "IC": 10,
    "P": 11,
    "CR": 12,
    "M": 13,
    "BTN": 14,
    "FB": 15,
    "CRA": 16,
    "SW": 17,
    "T": 18,
    "F": 19,
    "V": 20,
    "LED": 21,
    "S": 22,
    "QA": 23,
    "JP": 24
}

# Reverse mapping for convenience
int_to_class_name = {v: k for k, v in class_name_to_int.items()}

def map_classes_to_int(classes_list):
    return [class_name_to_int[cls] for cls in classes_list if cls in class_name_to_int]

@router.post("/predict-png")
async def predict_png(
    file: UploadFile = File(...),
    img_size: Optional[int] = Form(1280),
    show_conf: Optional[bool] = Form(True),
    show_labels: Optional[bool] = Form(True),
    show_boxes: Optional[bool] = Form(True),
    show_ocr: Optional[bool] = Form(True),
    line_width: Optional[int] = Form(None),
    classes: Optional[str] = Form([]),
):
    try:
        if line_width == None:
            line_width = img_size // 30

        # Read the uploaded file
        image = read_imagefile(await file.read())

        # Predict on the image
        if classes != []:
            results = model.predict(source=image, conf=0.3, iou=0.7, classes=map_classes_to_int(classes),)
        else:
            results = model.predict(source=image, conf=0.3, iou=0.7)
        # Extract prediction data
        img_with_boxes = results[0].plot(
            boxes=show_boxes,
            labels=show_labels,
            conf=show_conf,
            line_width=line_width,
        )
        if show_ocr:
            result = reader.readtext(image)
            for (bbox, text, prob) in result:
                top_left = tuple(map(int, bbox[0]))  # Convert [x1, y1] to integers
                bottom_right = tuple(map(int, bbox[2]))  # Convert [x3, y3] to integers
                img_with_boxes = cv2.rectangle(img_with_boxes, top_left, bottom_right, (0, 255, 0), 3)

        # Resize image
        resized_image = resize_image(img_with_boxes, img_size)

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

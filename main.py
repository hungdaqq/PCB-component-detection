from fastapi import (
    FastAPI,
    APIRouter,
    File,
    UploadFile,
    HTTPException,
    Response,
    Form,
    Depends,
)
from fastapi.responses import StreamingResponse
import uvicorn
from typing import Optional
import io
import csv
import cv2
import numpy as np
from ultralytics import YOLO
from io import BytesIO
from PIL import Image

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
    """
    Predict objects in an uploaded image using YOLO model.

    Args:
        file (UploadFile): The image file uploaded.

    Returns:
        dict: Dictionary containing predicted labels and bounding box coordinates.
    """
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
        print(results[0])
        return {
            "image_shape": {
                "height": results[0].orig_shape[0],
                "width": results[0].orig_shape[1],
            },
            "speed": results[0].speed,
            "predictions": predictions,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict-png")
async def predict(
    file: UploadFile = File(...),
    img_size: Optional[int] = Form(1280),
    show_conf: Optional[bool] = Form(True),
    show_labels: Optional[bool] = Form(True),
    show_boxes: Optional[bool] = Form(True),
    line_width: Optional[int] = Form(None),
):
    if line_width == None:
        line_width = img_size // 30
    try:
        # Read the uploaded file
        image = read_imagefile(await file.read())

        # Predict on the image
        results = model.predict(
            source=image,
            conf=0.3,
            iou=0.7,
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


app = FastAPI()
app.include_router(router)

if __name__ == "__main__":
    # Run the FastAPI app with Uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

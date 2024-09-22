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
from torchvision_detection import load_fasterrcnn, load_ssd, get_prediction
from utils import (
    class_name_to_int,
    color_values,
    int_to_class_name,
    map_classes_to_int,
    resize_image,
    read_imagefile,
)

# Initialize FastAPI app
router = APIRouter(prefix="/api/v1")


pcb_segmentation_model = YOLO("./weights/pcb_segmentation_best.pt")
pcb_detection_model = YOLO("./weights/pcb_detection_best.pt")
fasterrcnn_model = load_fasterrcnn(15)
ssd_model = load_ssd(15)


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


@router.post("/predict-faster-rcnn")
async def predict_ssd(
    file: UploadFile = File(...),
    img_size: Optional[int] = Form(1280),
    conf: Optional[float] = Form(0.25),
    # iou: Optional[float] = Form(0.45),
    show_conf: Optional[bool] = Form(True),
    # show_labels: Optional[bool] = Form(True),
    # show_boxes: Optional[bool] = Form(True),
    line_width: Optional[int] = Form(None),
    classes: Optional[str] = Form([]),
):
    if line_width == None:
        line_width = img_size // 30
    try:
        # Read the uploaded file
        image = read_imagefile(await file.read())

        boxes, labels, scores = get_prediction(
            fasterrcnn_model,
            image,
            conf,
            classes,
        )

        for box, label, score in zip(boxes, labels, scores):
            # Get bounding box coordinates
            xmin, ymin, xmax, ymax = map(int, box)

            # Get the class name and color
            class_name = int_to_class_name.get(label, "Unknown")
            color = color_values.get(
                label, (255, 255, 255)
            )  # Default to white if not found

            # Draw the bounding box
            cv2.rectangle(
                image,
                (xmin, ymin),
                (xmax, ymax),
                color,
                thickness=line_width,
            )
            if show_conf:
                label_text = f"{class_name}: {score:.2f}"
            else:
                label_text = class_name
            # Draw the label text above the bounding box
            cv2.putText(
                image,
                label_text,
                (xmin, ymin - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                4,  # Font scale
                color,
                img_size // 66,  # Line thickness
                cv2.LINE_AA,
            )

        # Resize image
        resized_image = resize_image(image, img_size)

        _, img_encoded = cv2.imencode(".png", resized_image)
        img_bytes = img_encoded.tobytes()

        return Response(content=img_bytes, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict-ssd")
async def predict_ssd(
    file: UploadFile = File(...),
    img_size: Optional[int] = Form(1280),
    conf: Optional[float] = Form(0.25),
    # iou: Optional[float] = Form(0.45),
    show_conf: Optional[bool] = Form(True),
    # show_labels: Optional[bool] = Form(True),
    # show_boxes: Optional[bool] = Form(True),
    line_width: Optional[int] = Form(None),
    classes: Optional[str] = Form([]),
):
    if line_width == None:
        line_width = img_size // 30
    try:
        # Read the uploaded file
        image = read_imagefile(await file.read())

        boxes, labels, scores = get_prediction(
            ssd_model,
            image,
            conf,
            classes,
        )

        for box, label, score in zip(boxes, labels, scores):
            # Get bounding box coordinates
            xmin, ymin, xmax, ymax = map(int, box)

            # Get the class name and color
            class_name = int_to_class_name.get(label, "Unknown")
            color = color_values.get(
                label, (255, 255, 255)
            )  # Default to white if not found

            # Draw the bounding box
            cv2.rectangle(
                image,
                (xmin, ymin),
                (xmax, ymax),
                color,
                thickness=line_width,
            )
            if show_conf:
                label_text = f"{class_name}: {score:.2f}"
            else:
                label_text = class_name
            # Draw the label text above the bounding box
            cv2.putText(
                image,
                label_text,
                (xmin, ymin - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                4,  # Font scale
                color,
                img_size // 66,  # Line thickness
                cv2.LINE_AA,
            )

        # Resize image
        resized_image = resize_image(image, img_size)

        _, img_encoded = cv2.imencode(".png", resized_image)
        img_bytes = img_encoded.tobytes()

        return Response(content=img_bytes, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    model: Optional[str] = Form("yolo"),
):
    if line_width == None:
        line_width = img_size // 30
    try:
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
            results = pcb_detection_model.predict(source=image, conf=conf, iou=iou)
        # Extract prediction data
        result_image = results[0].plot(
            boxes=show_boxes,
            labels=show_labels,
            conf=show_conf,
        )

        # Resize image
        resized_image = resize_image(result_image, img_size)

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
        print(map_classes_to_int(classes))
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
            result_image = cv2.addWeighted(
                image_rgb, 0.2, cv2.cvtColor(colored_mask, cv2.COLOR_BGR2RGB), 0.8, 0
            )

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

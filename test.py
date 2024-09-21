import cv2
import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np
from torchvision import transforms
import easyocr

# Load EasyOCR model
reader = easyocr.Reader(
    lang_list=["en"], gpu=True if torch.cuda.is_available() else False
)

# Load YOLOv8 model for text detection
text_detection_model = YOLO("./weights/text_detection_best.pt")

# Load your trained ResNet18 model for orientation detection
import torchvision.models as models

# Initialize the ResNet18 model architecture
orientation_model = models.efficientnet_b0(
    pretrained=False
)  # Initialize without pretrained weights
orientation_model.classifier[1] = torch.nn.Linear(1280, 4)
# Modify the final layer for 4 orientation classes (0, 90, 180, 270)

# Load the weights into the model
orientation_model.load_state_dict(torch.load("./weights/orientation_model.pth"))

# Set the model to evaluation mode
orientation_model.eval()

# Image transformations for ResNet18
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# Function to set image DPI and save it
def set_image_dpi(image_path, output_path, dpi=(70, 70)):
    img = Image.open(image_path)
    img.save(output_path, dpi=dpi)


# Function to predict orientation using ResNet18
def predict_orientation(image):
    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert image to PIL format for the transformation
    image_pil = Image.fromarray(image_rgb)

    # Apply transformations
    input_tensor = transform(image_pil).unsqueeze(0)  # Add batch dimension

    # Run the model and get the predicted orientation
    with torch.no_grad():
        output = orientation_model(input_tensor)
        _, predicted_label = torch.max(output, 1)  # Get the predicted class

    # Map the label to the correct rotation angle
    label_to_angle = {
        0: 0,
        1: 90,
        2: 180,
        3: 270,
    }  # Adjust this if your labels are different
    return label_to_angle[int(predicted_label)]


# Function to deskew image based on orientation prediction
def deskew_image(image_path):
    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]

    # Predict orientation using ResNet18
    rotation_angle = predict_orientation(image)
    print(image_path + ":", rotation_angle)

    # If rotation angle is not 0, rotate the image
    if rotation_angle != 0:
        # Calculate the rotation matrix
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, -rotation_angle, 1.0)

        # Calculate the new dimensions after rotation
        abs_cos = abs(M[0, 0])
        abs_sin = abs(M[0, 1])
        new_w = int(h * abs_sin + w * abs_cos)
        new_h = int(h * abs_cos + w * abs_sin)

        # Adjust the rotation matrix to account for the new size
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        # Perform the rotation
        image = cv2.warpAffine(
            image,
            M,
            (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    cv2.imwrite(image_path, image)


# Load the input image
image = cv2.imread("pcb_10f_cc_4.png")

# Perform YOLOv8 text detection
yolo_results = text_detection_model(image)

# Extract bounding boxes from YOLOv8 results
boxes = (
    yolo_results[0].boxes.xyxy.cpu().numpy()
)  # xyxy format (x_min, y_min, x_max, y_max)

for idx, box in enumerate(boxes):
    x_min, y_min, x_max, y_max = map(int, box[:4])

    # Crop the detected text area from the image
    cropped_image = image[y_min:y_max, x_min:x_max]

    # Save the cropped image for orientation detection
    cropped_image_path = f"./test/cropped_image_{idx}.png"
    cv2.imwrite(cropped_image_path, cropped_image)

    # Set DPI and save the cropped image with proper DPI
    dpi_cropped_image_path = f"./test/dpi_cropped_image_{idx}.png"
    set_image_dpi(cropped_image_path, dpi_cropped_image_path)

    # Deskew the cropped image using the ResNet18 orientation detection
    deskew_image(dpi_cropped_image_path)

    # Load the deskewed image
    deskewed_cropped_image = cv2.imread(dpi_cropped_image_path)

    # Convert deskewed image back to RGB for EasyOCR
    cropped_rgb = cv2.cvtColor(deskewed_cropped_image, cv2.COLOR_BGR2RGB)

    # Perform OCR on the deskewed cropped image using EasyOCR
    ocr_results = reader.readtext(cropped_rgb)

    # Draw the recognized text on the original image
    for detection in ocr_results:
        _, text, conf = detection

        # Draw text on the original image
        cv2.putText(
            image,
            text,
            (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )

# Save the final image with text and bounding boxes
cv2.imwrite("test_with_deskewed_text.png", image)

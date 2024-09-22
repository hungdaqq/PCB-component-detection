import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# Function to get predictions from the Faster R-CNN model
def get_prediction(model, img, threshold=0.5):
    # Transform image
    transform = T.Compose([T.ToTensor()])
    img = transform(img).unsqueeze(0)  # Add batch dimension

    # Get the predictions
    model.eval()
    with torch.no_grad():
        predictions = model(img)[0]

    # Filter out predictions below the threshold
    scores = predictions["scores"].numpy()
    keep = scores >= threshold
    boxes = predictions["boxes"][keep].cpu().numpy()
    labels = predictions["labels"][keep].cpu().numpy()
    scores = predictions["scores"][keep].cpu().numpy()

    return boxes, labels, scores


# Function to normalize RGB values to 0-1 range for matplotlib
def normalize_rgb(rgb_tuple):
    return tuple([x / 255.0 for x in rgb_tuple])


# Function to plot the predictions on the image with colors based on the class
def plot_predictions(img, boxes, labels, scores, class_name_to_int, color_values):
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # Reverse the dictionary to map class IDs to names
    int_to_class_name = {v: k for k, v in class_name_to_int.items()}

    # Plot each bounding box with the corresponding color
    for box, label, score in zip(boxes, labels, scores):
        # Create a rectangle patch for each bounding box
        xmin, ymin, xmax, ymax = box
        color = normalize_rgb(color_values[label])  # Normalize color
        rect = patches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )

        # Add the bounding box
        ax.add_patch(rect)

        # Add class name and confidence score
        class_name = int_to_class_name.get(label, "Unknown")
        plt.text(
            xmin,
            ymin,
            f"{class_name}: {score:.2f}",
            color="white",
            fontsize=12,
            bbox=dict(facecolor=color, edgecolor=color, alpha=0.5),
        )

    plt.show()


import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# Function to create the model architecture (based on your dataset's number of classes)
def load_model(num_classes):
    # Load a pre-trained Faster R-CNN model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=False, num_classes=num_classes
    )
    return model


model = load_model(15)

# Load your trained Faster R-CNN model (adjust the path to your model)
model.load_state_dict(torch.load("weights/faster-rcnn-ep30.pt", map_location="cuda"))
model.eval()

# Load an image
image_path = "pcb_10f_cc_4.png"
img = Image.open(image_path)

# Define your class_name_to_int mapping
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

# Define the color mapping for each class (as per your provided mapping)
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

# Get predictions
boxes, labels, scores = get_prediction(model, img)

# Plot the predictions
plot_predictions(img, boxes, labels, scores, class_name_to_int, color_values)

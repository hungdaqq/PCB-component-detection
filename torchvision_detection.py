import torch
from torchvision.models import detection
import torchvision.transforms as T
import numpy as np
import cv2
from utils import map_classes_to_int, resize_image


def load_fasterrcnn(num_classes, weight="weights/faster-rcnn-ep30.pt"):
    # Load a pre-trained Faster R-CNN model
    model = detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(weight, map_location="cpu"))
    model.eval()
    return model


def load_ssd(num_classes, weight="weights/ssp-ep30.pt"):
    # Load a pre-trained SSD model
    model = detection.ssd300_vgg16(pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(weight, map_location="cpu"))
    model.eval()
    return model


# Function to get predictions from the Faster R-CNN model
def get_prediction(model, image, conf=0.25, classes=[]):
    # image = resize_image(image, 640)
    # Transform image
    transform = T.Compose([T.ToTensor()])
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Send the image tensor to CPU
    image = image.cpu()

    # Get the predictions
    model.eval()
    with torch.no_grad():
        predictions = model(image)[0]

    # Filter out predictions below the threshold
    scores = predictions["scores"].cpu().numpy()  # Ensure scores are on CPU
    keep = scores >= conf
    boxes = predictions["boxes"][keep].cpu().numpy()  # Ensure boxes are on CPU
    labels = predictions["labels"][keep].cpu().numpy()  # Ensure labels are on CPU
    scores = predictions["scores"][keep].cpu().numpy()  # Ensure scores are on CPU
    # Filter by specific classes if provided
    print(boxes)
    if classes != []:
        keep_classes = np.isin(labels, map_classes_to_int(classes))
        boxes = boxes[keep_classes]
        labels = labels[keep_classes]
        scores = scores[keep_classes]

    return boxes, labels, scores

from torchvision.models.detection.ssd import SSDClassificationHead
import torchvision

model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
print(model)

import torch
import requests
import numpy as np
from PIL import Image
from io import BytesIO

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

"""
tried on efficient_b4 but just dumped astronomical amount of data to my gpu and turned off my 
system
so this efficeintnet_b0 seems to work fine (RTX 3050 6GB)
"""
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
# preloaded weights, we mght need to train it
weights = EfficientNet_B0_Weights.DEFAULT
model = efficientnet_b0(weights=weights).eval()

target_layer = model.features[-1] 

# some random image
url = "https://hips.hearstapps.com/hmg-prod/images/golden-retriever-royalty-free-image-506756303-1560962726.jpg"
response = requests.get(url)

rgb_img = Image.open(BytesIO(response.content)).convert('RGB')
# downsize to 256,256 or else system will just suicide
rgb_img.thumbnail((256, 256)) 
input_tensor = preprocess_image(np.array(rgb_img), mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225])

# ----- model prediction ----- 
output = model(input_tensor)
class_index = output.argmax(axis=1).item()
class_name = weights.meta["categories"][class_index]
print(f"Model prediction: {class_name} (Class Index: {class_index})")

# traget for gradcam
targets = [ClassifierOutputTarget(class_index)]

# gradcam heatmap
with GradCAM(model=model, target_layers=[target_layer]) as cam:
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    #  normalize it
    rgb_img_float = np.float32(rgb_img) / 255
    visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)

# heatmap overlayed on image
Image.fromarray(visualization).show()
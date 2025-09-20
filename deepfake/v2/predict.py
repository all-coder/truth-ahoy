import torch
import timm
import torch.nn as nn
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image

# Grad-CAM imports
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Load model ---
model_infer = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=2)
model_infer.load_state_dict(torch.load('model_weights.pth', map_location=device))
model_infer.to(device)
model_infer.eval()

# --- Preprocess a test image ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
])

img_path = "test.jpg"   # your test image path
pil_img = Image.open(img_path).convert('RGB')
img_tensor = transform(pil_img).unsqueeze(0).to(device)

# For visualization: denormalize to [0,1] RGB
img_for_cam = np.array(pil_img.resize((224,224))) / 255.0

# For transformation: preprocessing image
def vit_reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(-1))
    result = result.permute(0, 3, 1, 2)  #shape: [B, C, H, W]
    return result

# --- Prediction ---
logits = model_infer(img_tensor)
probs = torch.softmax(logits, dim=1)
predicted_class = logits.argmax(dim=1).item()
print(f"Predicted class: {predicted_class} | probs: {probs.detach().cpu().numpy()}")

# --- Grad-CAM setup ---
target_layers = [model_infer.blocks[-1].norm1]   # last norm in final block
targets = [ClassifierOutputTarget(predicted_class)]

with GradCAM(model=model_infer, target_layers=target_layers, reshape_transform=vit_reshape_transform) as cam:
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0, :]

# --- Overlay heatmap ---
cam_image = show_cam_on_image(img_for_cam, grayscale_cam, use_rgb=True)
cv2.imwrite("gradcam_result.jpg", cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))

print("Grad-CAM saved → gradcam_result.jpg")

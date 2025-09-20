import torch
import timm
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import os

def infer(
    model_path: str, 
    image_path: str, 
    output_dir="."
):
    """
    As the name states, performs inference & gradcam visualization
    class constraints:
    fake = 0
    real = 1
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ['fake', 'real'] 
    
    print(f"Using device: {device}")
    if not os.path.exists(model_path):
        print(f"ERROR: Model weights not found at: {model_path}")
        return
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found at: {image_path}")
        return

    print(f"Loading model from {model_path}...")
    model = timm.create_model('efficientnet_b1', pretrained=False, num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    pil_image = Image.open(image_path).convert('RGB')
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
        predicted_class_idx = np.argmax(probabilities)
        predicted_class_name = class_names[predicted_class_idx]
        confidence = probabilities[predicted_class_idx]

    print(f"\nPrediction: '{predicted_class_name}' with {confidence:.2%} confidence.")
    print(f"All probabilities: Fake={probabilities[0]:.2%}, Real={probabilities[1]:.2%}")

    print("Generating Grad-CAM heatmap...")
    target_layers = [model.conv_head]
    
    with GradCAM(model=model, target_layers=target_layers) as cam:
        targets = [ClassifierOutputTarget(predicted_class_idx)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        rgb_img = np.array(pil_image.resize((224, 224))) / 255.0
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        vis_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
        output_filename = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{output_filename}_gradcam_{predicted_class_name}.png")
        cv2.imwrite(output_path, vis_bgr * 255)
        print(f"Grad-CAM heatmap saved to: {output_path}")


if __name__ == "__main__":
    MODEL_PATH = "./deepfake/model/50epochs_efficientnet_b1.pth"
    IMAGE_PATH = "./rishi_sunak.jpeg" 
    OUTPUT_DIR = "./inference_results"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    infer(MODEL_PATH, IMAGE_PATH, OUTPUT_DIR)

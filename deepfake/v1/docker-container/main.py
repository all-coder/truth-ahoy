from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import timm
import numpy as np
from torchvision import transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import cv2
import os
import io

# --- App ---
app = FastAPI(title="Deepfake Classifier API")

# --- Device & class info ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['fake', 'real']

# --- Load model once ---
MODEL_PATH = "./deepfake/model/50epochs_efficientnet_b1.pth"
model = timm.create_model('efficientnet_b1', pretrained=False, num_classes=len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# --- Transform ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Endpoints ---
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert('RGB')
        input_tensor = transform(pil_image).unsqueeze(0).to(device)

        # Prediction
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            pred_class = class_names[pred_idx]
            confidence = float(probs[pred_idx])

        # Grad-CAM
        target_layers = [model.conv_head]
        with GradCAM(model=model, target_layers=target_layers) as cam:
            targets = [ClassifierOutputTarget(pred_idx)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            rgb_img = np.array(pil_image.resize((224, 224))) / 255.0
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            vis_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)

        # Save Grad-CAM image
        output_dir = "./inference_results"
        os.makedirs(output_dir, exist_ok=True)
        output_filename = os.path.splitext(file.filename)[0]
        output_path = os.path.join(output_dir, f"{output_filename}_gradcam_{pred_class}.png")
        cv2.imwrite(output_path, vis_bgr * 255)

        return JSONResponse({
            "prediction": pred_class,
            "confidence": confidence,
            "all_probs": {"fake": float(probs[0]), "real": float(probs[1])},
            "gradcam_image_path": output_path
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

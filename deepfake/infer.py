import base64
import numpy as np
from google.cloud import aiplatform
from PIL import Image
import cv2
import io

# ---------------- CONFIG ----------------
PROJECT = "electric-armor-472709-p6"
REGION = "us-central1"
ENDPOINT_ID = "4191825408733216768"
INPUT_IMAGE_PATH = "test.jpg"
OUTPUT_HEATMAP_PATH = "heatmap.jpg"
# ----------------------------------------

def load_image_as_b64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def save_heatmap_from_b64(b64_str, output_path):
    img_data = base64.b64decode(b64_str)
    img_array = np.frombuffer(img_data, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    cv2.imwrite(output_path, img)
    print(f"Saved heatmap to {output_path}")

def main():
    # Initialize Vertex AI
    aiplatform.init(project=PROJECT, location=REGION)
    endpoint = aiplatform.Endpoint(endpoint_name=ENDPOINT_ID)

    # Prepare request
    img_b64 = load_image_as_b64(INPUT_IMAGE_PATH)
    instances = [{"image_bytes": {"b64": img_b64}}]

    # Send request
    prediction = endpoint.predict(instances=instances)

    # Print outputs
    print("Prediction response:", prediction)

    # If your container returns Grad-CAM heatmap
    if "heatmap" in prediction[0].keys():
        heatmap_b64 = prediction[0]["heatmap"]
        save_heatmap_from_b64(heatmap_b64, OUTPUT_HEATMAP_PATH)

if __name__ == "__main__":
    main()

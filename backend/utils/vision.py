import base64
from google.cloud import vision
from helpers import get_service_key

# loading the google cloud service key
GOOGLE_CLOUD_SERVICE_KEY = get_service_key()
client = vision.ImageAnnotatorClient.from_service_account_json(GOOGLE_CLOUD_SERVICE_KEY)

# it is not a complete reverse image search as GCP services don't provide one, but it would give us leads and info, 
# which would enable us to conduct further analysis through web agent analysis.

# returns a dict containing where the image was used, and gives us the labels and relevant web entities.
def reverse_image_search(image_base64: str) -> dict:
    if "," in image_base64:
        image_base64 = image_base64.split(",")[1]
    content = base64.b64decode(image_base64)
    image = vision.Image(content=content)
    response = client.web_detection(image=image)
    annotations = response.web_detection

    if response.error.message:
        return {"error": response.error.message}

    result = {"best_guess_labels": [], "pages_with_matching_images": [], "web_entities": [], "visually_similar_images": []}

    if annotations.best_guess_labels:
        for label in annotations.best_guess_labels:
            result["best_guess_labels"].append(label.label)

    if annotations.pages_with_matching_images:
        for page in annotations.pages_with_matching_images:
            page_data = {"url": page.url, "full_matching_images": [], "partial_matching_images": []}
            if page.full_matching_images:
                for img in page.full_matching_images:
                    page_data["full_matching_images"].append(img.url)
            if page.partial_matching_images:
                for img in page.partial_matching_images:
                    page_data["partial_matching_images"].append(img.url)
            result["pages_with_matching_images"].append(page_data)

    if annotations.web_entities:
        for entity in annotations.web_entities:
            result["web_entities"].append({"score": entity.score, "description": entity.description})

    if annotations.visually_similar_images:
        for img in annotations.visually_similar_images:
            result["visually_similar_images"].append(img.url)

    return result

# detects landmarks if present in the image and returns the approximate location with latitude and longitude
def detect_landmarks(image_base64: str) -> dict:
    if "," in image_base64:
        image_base64 = image_base64.split(",")[1]
    content = base64.b64decode(image_base64)
    image = vision.Image(content=content)
    response = client.landmark_detection(image=image)
    annotations = response.landmark_annotations

    if response.error.message:
        return {"error": response.error.message}

    result = {"landmarks": []}

    if annotations:
        for landmark in annotations:
            landmark_data = {"description": landmark.description, "locations": []}
            if landmark.locations:
                for loc in landmark.locations:
                    lat_lng = loc.lat_lng
                    landmark_data["locations"].append({
                        "latitude": lat_lng.latitude,
                        "longitude": lat_lng.longitude
                    })
            result["landmarks"].append(landmark_data)

    return result

import os
import requests
import base64
from google.cloud import vision
from utils.helpers import get_service_key
from dotenv import load_dotenv
from google import genai
from utils.helpers import fetch_image_from_url
load_dotenv()

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

    result = {
        "best_guess_labels": [],
        "pages_with_matching_images": [],
        "web_entities": [],
        "visually_similar_images": [],
    }

    if annotations.best_guess_labels:
        for label in annotations.best_guess_labels:
            result["best_guess_labels"].append(label.label)

    if annotations.pages_with_matching_images:
        for page in annotations.pages_with_matching_images:
            page_data = {
                "url": page.url,
                "full_matching_images": [],
                "partial_matching_images": [],
            }
            if page.full_matching_images:
                for img in page.full_matching_images:
                    page_data["full_matching_images"].append(img.url)
            if page.partial_matching_images:
                for img in page.partial_matching_images:
                    page_data["partial_matching_images"].append(img.url)
            result["pages_with_matching_images"].append(page_data)

    if annotations.web_entities:
        for entity in annotations.web_entities:
            result["web_entities"].append(
                {"score": entity.score, "description": entity.description}
            )

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
                    landmark_data["locations"].append(
                        {"latitude": lat_lng.latitude, "longitude": lat_lng.longitude}
                    )
            result["landmarks"].append(landmark_data)

    return result


def reverse_geocoding(latitude: float, longitude: float, limit: int = 4):
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    GEOCODING_KEY = os.getenv("GEOCODING_API")
    params = {"latlng": f"{latitude},{longitude}", "key": GEOCODING_KEY}
    resp = requests.get(url, params=params)
    data = resp.json()
    result = {
        "possible_addresses": [
            {
                r["formatted_address"]: {
                    "location_type": r["geometry"]["location_type"],
                    "types": r.get("types", []),
                }
            }
            for r in data.get("results", [])[:limit]
        ],
        "country_code": data.get("plus_code", {}).get("compound_code"),
    }
    return result


### Image Analysis ###
def get_system_prompt(landmarks_context: str, only_claim: bool = False):
    if only_claim:
        extra_prompt = "Note: Only the claim image is available. Explain the content, possible context, and any suspicious elements based on this single image."
    else:
        extra_prompt = ""

    prompt = (
        "You are an assistant that performs deepfake and misinformation analysis. "
        "The first image provided is the claim/original image. The subsequent images "
        "are contexts or other usages of the same or similar image. "
        "Your task is to carefully compare the claim image with the others, analyze "
        "differences in content, manipulations, or alterations, and determine whether "
        "the claim image is authentic, misleading, out of context, or fake. "
        f"{landmarks_context} {extra_prompt} "
        "Highlight differences, whether the contexts match, and provide a clear explanation."
    )
    return prompt


def get_deepfake_system_prompt(classification: str):
    inconsistency_list = """
        - Inconsistent or blurry object boundaries
        - Discontinuous or warped surfaces
        - Misaligned facial features (eyes, mouth, ears)
        - Asymmetric or unnatural expressions
        - Floating or disconnected elements
        - Irregular proportions in the face or body
        - Non-manifold geometries in rigid objects
        - Glitches in hair, teeth, or glasses
        - Inconsistent lighting or shadows
        - Blurred or mismatched textures
        - Issues around jewelry, hats, or accessories
        - Strange reflections in eyes or glasses
        - Background inconsistencies or distortions
        - Pixelation or compression anomalies"""

    prompt = f"""You are an assistant performing deepfake and misinformation analysis. 
        First, determine whether the claim image is suitable for deepfake classification. 
        Consider if it might be a setting image, a copy-move image, a reused background, or any other scenario that could make classification unreliable. 
        Explain your reasoning and any limitations in detecting manipulation.

        The claim image and its Grad-CAM mask are provided, along with its classification: {classification}. 
        Use the Grad-CAM to focus on relevant regions and analyze whether the image appears authentic or manipulated. 
        Explain only the applicable inconsistencies from the list below, supporting why the classification is justified or not. 
        Limit explanations to 50 words per inconsistency. Do not format the text.

        Inconsistencies to be considered:
        {inconsistency_list}"""

    return prompt

# dummy function for now
def predict_deepfake():
    pass

def deepfake_image_analysis(image_b64: str = None):
    if image_b64 is None:
        return "No image provided. Deepfake analysis was not performed."
    try:
        classification, claim_b64, gradcam_b64 = predict_deepfake()
        system_instruction = get_deepfake_system_prompt(classification)
        analysis_output = gemini_image_analysis(images_b64=[claim_b64, gradcam_b64], system_instruction=system_instruction)
        if analysis_output is None:
            return "Deepfake analysis could not be performed on the image."
    except Exception as e:
        return "Deepfake analysis failed for the image."

def deep_image_analysis(claim_image_url: str = None):
    if claim_image_url is None:
        return "No image provided. Deepfake analysis was not performed."

    try:
        _,claim_image_b64 = fetch_image_from_url(claim_image_url)
        reverse_search = reverse_image_search(claim_image_b64)
        landmarks_search = detect_landmarks(claim_image_b64)

        label = reverse_search.get("best_guess_labels", ["No Labels"])[0]
        print(f"Reverse image search completed, best guess label: {label}")

        landmarks = landmarks_search.get("landmarks")
        if not landmarks:
            landmark_context = "No Landmarks"
        else:
            landmark_context = "Possible locations of the image:\n\n"
            locations = [(loc["latitude"], loc["longitude"]) for loc in landmarks["locations"]][:4]
            for lat, lng in locations:
                reverse_geocode = reverse_geocoding(lat, lng)
                landmark_context += reverse_geocode + "\n\n"
        print("Landmark context prepared")

        image_links = [
            img
            for page in reverse_search.get("pages_with_matching_images", [])[:2]
            for img in page.get("partial_matching_images", [])
        ]
        page_links = [
            page["url"]
            for page in reverse_search.get("pages_with_matching_images", [])[:2]
            for img in page.get("partial_matching_images", [])
        ]
        print(f"Image and page links collected: {len(image_links)} images, {len(page_links)} pages")

        image_links_b64 = []
        for url in image_links:
            _, img_b64 = fetch_image_from_url(url)
            image_links_b64.append(img_b64)
        image_links_b64 = image_links_b64[:2]
        print("Fetched images converted to base64")

        only_claim_image = len(image_links_b64) == 0
        system_prompt = get_system_prompt(landmark_context, only_claim=only_claim_image)
        image_b64_to_query = [claim_image_b64] + image_links_b64[:1]

        response_text = gemini_image_analysis(images_b64=image_b64_to_query, system_instruction=system_prompt)
        print("Analysis completed")

        return response_text

    except Exception as e:
        return f"Error during analysis: {e}"


def gemini_image_analysis(
    images_b64: list[str] = None,
    system_instruction: str = "",
    model: str = "gemini-1.5-flash",
):  
    # we will keep a separate gemini API for image analysis
    api_key = os.getenv("GEMINI_IMAGE")
    client = genai.Client(api_key=api_key)

    contents = [system_instruction] if system_instruction else []

    if images_b64:
        for img_b64 in images_b64:
            img_obj = {
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": base64.b64decode(img_b64),
                }
            }
            contents.append(img_obj)

    response = client.models.generate_content(model=model, contents=contents)
    return response.text

import os
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from dotenv import load_dotenv


load_dotenv()

API_KEY = os.getenv("ROBOFLOW_API_KEY")
if not API_KEY:
    raise ValueError("Missing ROBOFLOW_API_KEY in environment variables.")

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY
)


def detect_object(image_path: str, model_id: str = 'nutrition-label-table/3'):
    image = Image.open(image_path)

    result = CLIENT.infer(image_path, model_id=model_id)

    if not result["predictions"]:
            print("No predictions found.")
            return None

    top_pred = max(result["predictions"], key=lambda x: x["confidence"])

    x = top_pred["y"]
    y = top_pred["x"]
    w = top_pred["width"]
    h = top_pred["height"]

    left = int(x  - w / 2)
    top = int(y  - h / 2)
    right = int(x + w / 2)
    bottom = int(y + h / 2)

    cropped = image.crop((left, top, right, bottom))
    cropped_display = ImageOps.exif_transpose(cropped) # Fix orientation for image with EXIF tag

    plt.figure()
    plt.imshow(cropped_display)
    plt.axis("off")
    plt.show()
    
    return cropped_display

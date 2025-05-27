import cv2
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model(num_classes: int = 2, model_path: str = "./model/best_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return device, model

def detect_object(image_path: str, show_result: bool = True):

    device, model = get_model(num_classes=2, model_path="./model/best_model.pth")

    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (640, 640))
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)

    pil_image = Image.fromarray(rgb_image)
    img_tensor = T.ToTensor()(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(img_tensor)[0]

    boxes = prediction["boxes"]
    scores = prediction["scores"]

    cropped_result = None
    for box, score in zip(boxes, scores):
        if score < 0.5:
            continue
        xmin, ymin, xmax, ymax = box.int().cpu().numpy()
        cropped_result = resized_image[ymin:ymax, xmin:xmax]

        if show_result is True and cropped_result is not None:
            plt.imshow(cropped_result, cmap="gray")
            plt.title(f"Confidence: {score:.2f}")
            plt.axis("off")
            plt.show()

        break

    if cropped_result is not None:
        return cv2.cvtColor(cropped_result, cv2.COLOR_GRAY2BGR)
    
    return None

from collections import Counter
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

app = FastAPI(title="Insect Detection API")

MODEL_PATH = Path(__file__).parent / "exp-6.pt"
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = YOLO(str(MODEL_PATH))

# Read the three class names from your trained model.
CLASS_NAMES = model.names
print("Model classes:", CLASS_NAMES)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "classes": CLASS_NAMES
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image.")

    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid image.")

    results = model.predict(image, conf=0.25, verbose=False)
    result = results[0]

    counts = Counter()
    detections = []

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            class_name = CLASS_NAMES[class_id]
            counts[class_name] += 1

            detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "bbox": [
                    round(x1, 2),
                    round(y1, 2),
                    round(x2, 2),
                    round(y2, 2)
                ]
            })

    # Include all model classes, even when their count is zero.
    class_counts = {
        str(name): counts.get(name, 0)
        for name in CLASS_NAMES.values()
    }

    return {
        "image_width": image.width,
        "image_height": image.height,
        "detections": detections,
        "counts": class_counts,
        "total": len(detections)
    }

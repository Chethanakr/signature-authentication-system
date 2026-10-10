import io
import cv2
import traceback
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from model import SiameseResNet18

# Global state holders
device = None
model = None
OPTIMAL_THRESHOLD = 1.3671

# 1. Update normalization for 1-channel grayscale input
transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

@asynccontextmanager
async def lifespan(app: FastAPI):
    global device, model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SiameseResNet18(embedding_dim=128).to(device)
    weights_path = "weights/best_siamese_resnet18.pth"
    
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        print(f"Model loaded successfully on {device}.")
    except Exception as e:
        print(f"Failed to load weights from {weights_path}: {e}")

    yield

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(title="Signature Verification API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_signature(image_bytes: bytes, filename: str = "") -> torch.Tensor:
    if filename.lower().endswith(".pdf"):
        raise ValueError("PDF files are not supported. Please upload an image format (PNG, JPG, JPEG).")

    pil_img = Image.open(io.BytesIO(image_bytes)).convert('L')
    img_np = np.array(pil_img)

    if img_np.size == 0:
        raise ValueError("Uploaded image file is empty or corrupted.")

    blurred = cv2.GaussianBlur(img_np, (5, 5), 0)
    _, binary_img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    coords = cv2.findNonZero(binary_img)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        pad = 10
        h_img, w_img = binary_img.shape
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(w_img, x + w + pad), min(h_img, y + h + pad)
        
        if (x2 - x1) > 5 and (y2 - y1) > 5:
            cropped = img_np[y1:y2, x1:x2]
        else:
            cropped = img_np
    else:
        cropped = img_np

    # 2. Convert to single-channel Grayscale ('L') to match model conv1 weights
    cleaned_pil = Image.fromarray(cropped).convert('L')
    return transform_pipeline(cleaned_pil).unsqueeze(0).to(device)

@app.post("/api/v1/verify")
async def verify_signatures(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    try:
        bytes1 = await file1.read()
        bytes2 = await file2.read()

        t1 = preprocess_signature(bytes1, file1.filename)
        t2 = preprocess_signature(bytes2, file2.filename)

        with torch.no_grad():
            out1, out2 = model(t1, t2)
            out1 = F.normalize(out1, p=2, dim=1)
            out2 = F.normalize(out2, p=2, dim=1)
            distance = F.pairwise_distance(out1, out2).item()

        is_genuine = distance < OPTIMAL_THRESHOLD

        return {
            "distance": round(distance, 4),
            "threshold": OPTIMAL_THRESHOLD,
            "is_genuine": is_genuine,
            "verdict": "GENUINE MATCH" if is_genuine else "FORGERY DETECTED"
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))
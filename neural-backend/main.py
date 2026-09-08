import io
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from model import SiameseResNet18

app = FastAPI(title="Signature Verification API")

# Allow requests from Vite frontend (localhost:5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SiameseResNet18(embedding_dim=128).to(device)

WEIGHTS_PATH = "weights/best_siamese_resnet18.pth"
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.eval()

OPTIMAL_THRESHOLD = 1.3671

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

@app.post("/api/v1/verify")
async def verify_signatures(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    try:
        bytes1 = await file1.read()
        bytes2 = await file2.read()

        img1 = Image.open(io.BytesIO(bytes1)).convert('L')
        img2 = Image.open(io.BytesIO(bytes2)).convert('L')

        t1 = transform(img1).unsqueeze(0).to(device)
        t2 = transform(img2).unsqueeze(0).to(device)

        with torch.no_grad():
            out1, out2 = model(t1, t2)
            distance = F.pairwise_distance(out1, out2).item()

        is_genuine = distance < OPTIMAL_THRESHOLD

        return {
            "distance": round(distance, 4),
            "threshold": OPTIMAL_THRESHOLD,
            "is_genuine": is_genuine,
            "verdict": "GENUINE MATCH" if is_genuine else "FORGERY DETECTED"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
import cv2

app = FastAPI()

# Permitir CORS (Kotlin app pode consumir)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # depois pode restringir
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("runs/detect/train/weights/best.pt")
classes = ['1', 'Capacete', 'Detector_Tensao_Contato', 'Pessoa', 'Pessoa-Sem Capacete-', 'Vara_Manobra']

@app.post("/detectar/")
async def detectar(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(image)
    detectados = [classes[int(cls)] for cls in results[0].boxes.cls]

    return JSONResponse(content={"detectados": detectados})


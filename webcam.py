from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import io

# Criação do app FastAPI
app = FastAPI()

# CORS para permitir chamadas do seu app Kotlin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Você pode restringir isso depois para o domínio do app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carregamento do modelo YOLO
model = YOLO("runs/detect/train/weights/best.pt")
classes = ['1', 'Capacete', 'Detector_Tensao_Contato', 'Pessoa', 'Pessoa-Sem Capacete-', 'Vara_Manobra']

@app.post("/detectar")
async def detectar_imagem(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Realiza predição
        results = model(image)
        detected_classes = [classes[int(c)] for r in results for c in r.boxes.cls]

        # Exemplo simples de regra de alerta
        if "Pessoa" in detected_classes and "Capacete" in detected_classes:
            status = "✅ Pessoa com capacete detectada"
        elif "Pessoa" in detected_classes and "Capacete" not in detected_classes:
            status = "⚠️ Pessoa sem capacete detectada"
        else:
            status = "🔍 Nenhuma pessoa detectada"

        return JSONResponse(content={
            "status": status,
            "detected": detected_classes
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

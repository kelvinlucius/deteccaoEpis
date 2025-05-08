from ultralytics import YOLO
import cv2
import threading
import pygame

# Carrega o modelo treinado
model = YOLO("runs/detect/train/weights/best.pt")

# Lista de classes (deve bater com as IDs do seu modelo)
classes = ['1', 'Capacete', 'Detector_Tensao_Contato', 'Pessoa', 'Pessoa-Sem Capacete-', 'Vara_Manobra']

# Função para tocar o alerta positivo
def tocar_sucesso():
    pygame.mixer.init()
    pygame.mixer.music.load("ok.mp3")  # ou .wav
    pygame.mixer.music.play()

# Inicia a webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Faz a predição
    results = model(frame)
    annotated_frame = results[0].plot()

    # Pega todas as classes detectadas no frame
    detected_classes = [classes[int(cls)] for cls in results[0].boxes.cls]

    # Verifica se ambas "Pessoa" e "Capacete" foram detectadas
    if "Pessoa" in detected_classes and "Capacete" in detected_classes:
        print("✅ Pessoa com capacete detectada, pronta para trabalhar!")
        threading.Thread(target=tocar_sucesso).start()
    elif "Pessoa" in detected_classes and "Pessoa-Sem Capacete-" in detected_classes:
        print("⚠️ Pessoa sem capacete detectada!")

    # Exibe o frame anotado
    cv2.imshow("Detecção com Alerta", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

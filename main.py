import cv2
from ultralytics import YOLO

# Carrega o modelo YOLOv8 (pode usar um customizado depois)
model = YOLO('yolov8n.pt')  # Modelo leve padrão

# Abre a webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detecção
    results = model(frame)

    # Exibir com boxes desenhados
    annotated_frame = results[0].plot()
    cv2.imshow("Detecção de EPIs", annotated_frame)

    # Tecla 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

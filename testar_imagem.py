from ultralytics import YOLO
import cv2

# Carrega o modelo
model = YOLO("runs/detect/train/weights/best.pt")

# Caminho da imagem de teste
image_path = "contrucao.jpg"  # Substitua pelo nome da imagem

# Faz a detecção
results = model(image_path)

# Exibe a imagem anotada
annotated = results[0].plot()
cv2.imshow("Resultado", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

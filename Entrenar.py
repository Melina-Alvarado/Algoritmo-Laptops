from ultralytics import YOLO

#Cargar modelo
model = YOLO("yolo11n.pt")

#Entrenar Modelo
results = model.train(data="dataset.yaml", epochs=100, imgsz=640)
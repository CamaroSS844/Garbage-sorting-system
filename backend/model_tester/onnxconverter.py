from ultralytics import YOLO

# Load the model properly
model = YOLO(r"C:\Users\Taboka\Downloads\85 epochs\my_model (3)\train\weights\best.pt")

# Export to ONNX
model.export(format="onnx")
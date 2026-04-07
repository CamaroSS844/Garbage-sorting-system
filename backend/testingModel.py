import cv2
from ultralytics import YOLO

# Load your model
model = YOLO(r'C:\Users\Taboka\Documents\others\personal final project\my_model\my_model.pt')

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run inference on the frame
    results = model(frame)  # same as model.predict(frame)

    # Get the frame with bounding boxes and labels drawn
    annotated_frame = results[0].plot()

    # Show the frame
    cv2.imshow('YOLO Inference', annotated_frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
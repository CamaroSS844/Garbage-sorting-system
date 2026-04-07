import onnxruntime as ort
import cv2
import numpy as np
import time
 
# Load ONNX model
session = ort.InferenceSession("C:\\Users\\Taboka\\Documents\\others\\personal final project\\openvino model\\my_model.onnx", providers=['CUDAExecutionProvider'])
input_name = session.get_inputs()[0].name
 
# Read video
video_path = r"C:\Users\Taboka\Downloads\gettyimages-1212816860-640_adpp.mp4"
cap = cv2.VideoCapture(video_path)
 
# get width and height
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
 
# FPS calculation variables
fps_counter = 0
fps_start_time = time.time()
fps_display = 0.0
 
# Load COCO class names from file
with open('C:\\Users\\Taboka\\Documents\\others\\personal final project\\backend\\model_tester\\classes.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]
 
def postprocess_yolo_output(outputs, original_shape, conf_threshold=0.3, iou_threshold=0.45):
    """Post-process YOLOv8 ONNX output"""
    predictions = outputs[0]  # Shape: [1, 84, 8400]
    predictions = predictions[0]  # Remove batch dimension: [84, 8400]
    predictions = predictions.T  # Transpose to [8400, 84]
     
    # Extract boxes and scores
    boxes = predictions[:, :4]  # First 4 columns are bbox coordinates
    scores = predictions[:, 4:]  # Remaining columns are class scores
     
    # Get the class with highest score for each detection
    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)
     
    # Filter by confidence threshold
    valid_detections = confidences > conf_threshold
    boxes = boxes[valid_detections]
    confidences = confidences[valid_detections]
    class_ids = class_ids[valid_detections]
     
    # Convert from center format to corner format
    x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
     
    boxes = np.column_stack((x1, y1, x2, y2))
     
    # Scale boxes to original image size
    orig_h, orig_w = original_shape[:2]
    boxes[:, [0, 2]] *= orig_w / 640  # Scale x coordinates
    boxes[:, [1, 3]] *= orig_h / 640  # Scale y coordinates
 
    # Apply Non-Maximum Suppression to eliminate duplicate detections
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), conf_threshold, iou_threshold)
 
    # Check if any boxes remain after NMS
    if len(indices) > 0:
        indices = indices.flatten()
        return boxes[indices], confidences[indices], class_ids[indices]
    else:
        return [], [], []
 
# Process video frames
while cap.isOpened():
    frame_start_time = time.time()
     
    ret, frame = cap.read()
    if not ret:
        break
 
    # YOLOv8 preprocessing
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (640, 640))
    img_resized = img_resized.astype(np.float32) / 255.0
    img_resized = np.transpose(img_resized, (2, 0, 1))  # HWC -> CHW
    img_resized = np.expand_dims(img_resized, axis=0)
 
    # Run inference
    outputs = session.run(None, {input_name: img_resized})
 
    # Post-process outputs
    boxes, confidences, class_ids = postprocess_yolo_output(outputs, frame.shape)
 
    # Draw detections
    for box, conf, cls_id in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = map(int, box)
         
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
         
        # Draw label
        label = f"{class_names[cls_id]}: {conf:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
 
    # Calculate and display FPS
    fps_counter += 1
    if fps_counter % 10 == 0:  # Update FPS display every 10 frames
        fps_end_time = time.time()
        fps_display = 10 / (fps_end_time - fps_start_time)
        fps_start_time = fps_end_time
     
    # Draw FPS
    fps_text = f"FPS: {fps_display:.1f}"
    cv2.rectangle(frame, (5, height - 40), (120, height - 10), (0, 0, 0), -1)
    cv2.putText(frame, fps_text, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
 
    # Display frame
    cv2.imshow("YOLOv8 ONNX Runtime GPU", frame)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# Release everything
cap.release()
cv2.destroyAllWindows()

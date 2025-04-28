from ultralytics import YOLO

model = YOLO('yolov8n.pt')


# Run inference on the image
results = model('OptiLine/data/frames/seq_000062.jpg')  # Results object

# Save the annotated image
path = 'OptiLine/outputs/labeled_ims/output.jpg'  # Or whatever path you want

# results[0] corresponds to the first (and only) image
annotated_frame = results[0].plot()  # This draws boxes and labels

# Save it using OpenCV
import cv2
cv2.imwrite(path, annotated_frame)

print(f"Saved annotated image to {path}")
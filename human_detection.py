from ultralytics import YOLO
from CameraFeed import CameraFeed
import numpy as np

model = YOLO('yolov8n.pt')

EXPECTED_IMAGE_SIZE = (480, 640, 3)

# Mask out everythin except right-most pathway
mask = np.zeros(EXPECTED_IMAGE_SIZE)
mask[:, EXPECTED_IMAGE_SIZE[1] * 3 // 5 : , :] = 1

feed = CameraFeed(r'OptiLine/data/frames', skip_to= 1929, mask=mask)

i = 0
while True:
    try:
        # Run inference on the image
        results = model(next(feed))  # Results object
    except StopIteration:
        break

    # Save the annotated image
    path = f'OptiLine/outputs/labeled_ims/output_{i}.jpg'  

    # results[0] corresponds to the first (and only) image
    annotated_frame = results[0].plot()  # This draws boxes and labels

    # Save it using OpenCV
    import cv2
    cv2.imwrite(path, annotated_frame)

    i += 1
from CameraFeed import CameraFeed
from Annotator import Annotator
import numpy as np

EXPECTED_IMAGE_SIZE = (480, 640, 3)
# Mask out everythin except right-most pathway
mask = np.zeros(EXPECTED_IMAGE_SIZE)
mask[:, EXPECTED_IMAGE_SIZE[1] * 3 // 5 : , :] = 1

feed = CameraFeed(r'OptiLine/data/frames', skip_to= 1929, mask=mask)
annotator = Annotator()

i = 0
while True:
    try:
        im = next(feed)
    except StopIteration:
        break

    annotated_frame = annotator(im)

    # Save the annotated image
    path = f'OptiLine/outputs/labeled_ims/output_{i}.jpg'  

    # Save it using OpenCV
    import cv2
    cv2.imwrite(path, annotated_frame)

    i += 1
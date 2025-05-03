from CameraFeed import CameraFeed
from Annotator import Annotator
import numpy as np

EXPECTED_IMAGE_SIZE = (480, 640, 3)

# Want to keep foreground on right pathway
mask = np.zeros(EXPECTED_IMAGE_SIZE)
mask[EXPECTED_IMAGE_SIZE[0] * 2 // 5 :  , # Keep bottom 2/5
     EXPECTED_IMAGE_SIZE[1] * 3 // 5 : , :] = 1 # Keep right 2/5

skip_to = 1929
skip_to = 1700
# mask = None

feed = CameraFeed(r'OptiLine/data/frames', 
                  skip_to= skip_to, 
                  mask=mask, 
                  expected = EXPECTED_IMAGE_SIZE, )

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
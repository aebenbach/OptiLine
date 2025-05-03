import numpy as np 
import cv2 

vchr = np.vectorize(chr)

def label_vec(n): # assumes k <= 26
    if n == 0:
        return np.array([])

    k = (n // 26 ) + 1

    a = np.arange(26)

    labs = vchr((a % 26) + 65) 

    if (k > 1) and (k <= 26):
        kry = np.arange(k)
        labs2 = vchr((kry % 26) + 65)

        labs = np.char.add(labs2[:,None], labs[None,:])



    # ary = np.arange(n*k).reshape((k,n))

    # labs = vchr((ary % 26) + 65) 

    # if (k > 1) and (k <= 26):
    #     kry = np.arange(k)
    #     labs2 = vchr((kry % 26) + 65)

    #     labs = np.char.add(labs2, labs)
    
    # elif k > 26:
    #     raise Exception
    
    
    return labs.flatten()


class Annotator:
    def __init__(self, yolo_model = 'yolov8n.pt'):
        from ultralytics import YOLO

        self.model = YOLO(yolo_model)

    def __call__(self, im):
        return self.annotate(im)
    
    def annotate(self, im):
        results = self.model(im)  # YOLO model inference
        boxes = results[0].boxes  # Get bounding boxes

        human_boxes = boxes[boxes.cls == 0]

        results[0].boxes = human_boxes

        # Convert image to OpenCV format
        annotated_frame = im.copy()
        if not isinstance(annotated_frame, np.ndarray):
            annotated_frame = results[0].orig_img.copy()
        
        labs = label_vec(len(human_boxes))

        for idx, box in enumerate(human_boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # bounding box coordinates
            label = labs[idx]

            # Draw the bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw the label
            # print(labs, label)
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return annotated_frame




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

        annotated_frame = results[0].plot(
                conf=False,         
                labels=False,       
                boxes=True,        
                probs=False,
        )

        return annotated_frame



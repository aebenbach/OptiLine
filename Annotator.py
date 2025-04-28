class Annotator:
    def __init__(self, yolo_model = 'yolov8n.pt'):
        from ultralytics import YOLO

        self.model = YOLO(yolo_model)

    def __call__(self, im):
        return self.annotate(im)
    
    def annotate(self, im):
        results = self.model(im)

        annotated_frame = results[0].plot() 

        return annotated_frame



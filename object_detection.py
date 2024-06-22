import torch

class ObjectDetection:
    def __init__(self, model_path='yolov5s.pt'):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    
    def detect_objects(self, frame, min_confidence=0.75):
        results = self.model(frame)

        # # Filter detections based on confidence score
        # filtered_results = []
        # for det in results.xyxy[0]:
        #     if det[4] >= min_confidence:
        #         filtered_results.append(det)

        return results

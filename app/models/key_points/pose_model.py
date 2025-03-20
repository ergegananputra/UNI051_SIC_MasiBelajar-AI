
import cv2
from ultralytics import YOLO


class PoseModel:
    def __init__(self, weight_path: str = 'app/models/key_points/config/yolo11m-pose.pt'):
        self.weight_path = weight_path
        self.model : YOLO = YOLO(self.weight_path)

    def inference(self, inference_path: str, **kwargs):
        return self.model(inference_path, **kwargs)
    

    def stream_webcam(self):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise ValueError("Unable to open video source")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            result = self.inference(frame)

            # Perform pose detection on the current frame
            results = self.model.predict(frame)

            # Draw the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("Pose Detection", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the webcam and close OpenCV windows
        cap.release()
        cv2.destroyAllWindows()
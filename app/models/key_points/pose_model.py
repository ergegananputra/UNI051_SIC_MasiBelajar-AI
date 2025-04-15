import cv2
import numpy as np
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

    def is_fall(self, keypoint: np.ndarray):
        # Upper Body Threshold 6
        # Middle Body Threshold 10

        score = {
            "UpperBody": 0,
            "MiddleBody": 0,
            "LowerBody": 0,
        }

        # Fast calculate using numpy tensor and validate with conf > 0.5
        keypoint = keypoint[keypoint[:, 2] > 0.5]
        upper = keypoint[:6, 1] 
        middle = keypoint[6:10, 1]
        lower = keypoint[10:, 1]
        score["UpperBody"] = np.average(upper)
        score["MiddleBody"] = np.average(middle)
        score["LowerBody"] = np.average(lower)

    
        if score["UpperBody"] > score["LowerBody"]:
            return True
        return False

    def analyze_video(self, inference_path: str):
        """
        Analyze video and stream the results to Streamlit.
        
        :param inference_path: Path to the video file.
        """
        cap = cv2.VideoCapture(inference_path)
        if not cap.isOpened():
            raise ValueError("Unable to open video source:", inference_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform pose detection on the current frame
            result = self.model.predict(frame)[0]
            boxes = result.boxes.cpu().numpy()
            keypoints = result.keypoints
            keypoints = keypoints.data.cpu().numpy()

            frame = result.plot(
                line_width=1,
                labels=False,
            )

            # Draw the keypoints and bounding boxes on the frame
            for i, keypoint in enumerate(keypoints):
                # raise ValueError(f"Size: {keypoint.shape} \nkeypoint: {keypoint}")
                is_fall = self.is_fall(keypoint)


                # draw bounding box and detect if the person is falling
                try:
                    x_min, y_min, x_max, y_max = boxes.xyxy[i]
                except IndexError:
                    print(f"IndexError: {i} out of range for boxes.xyxy")
                if is_fall:
                    # cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)
                    cv2.putText(frame, "Falling", (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    # cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
                    cv2.putText(frame, "Standing", (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    
        




            # Display the annotated frame
            cv2.imshow("Pose Detection", frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
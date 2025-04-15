import cv2
import numpy as np
from shapely import Polygon
from ultralytics import YOLO


class SafezoneModel:
    def __init__(self, weight_path: str = 'app/models/object_detection/config/weight.pt'):
        self.weight_path = weight_path
        self.model : YOLO = YOLO(self.weight_path)

    def predict_object_detection(self, inference_path: str, **kwargs):
        return self.model(inference_path, **kwargs)
    
    def process_frame(self, frame, points: list):
        """
        Process a single frame for object detection and return the processed frame.
        """
        result = self.predict_object_detection(frame)

        bounding_boxes = result[0].boxes.xyxy.cpu().numpy()
        confidences = result[0].boxes.conf.cpu().numpy()
        classes = result[0].boxes.cls.cpu().numpy()
        class_names = result[0].names

        for i, box in enumerate(bounding_boxes):
            x_min, y_min, x_max, y_max = box[:4]
            confidence = confidences[i]
            class_id = int(classes[i])


            if class_names[class_id] == 'toddler':
                # Check how many area percentage of the bounding box is inside the safezone
                bbox = Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])
                safezone = Polygon(points)
                intersection = safezone.intersection(bbox)
                intersection_area = intersection.area
                bbox_area = bbox.area

                if intersection_area == bbox_area:
                    area_percentage = 1
                else:
                    area_percentage = intersection_area / bbox_area

                if area_percentage >= 0.5:
                    color=(0, 255, 0)
                else:
                    color = (0, 0, 255)

                cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color=color, thickness=2)

                label = f"{int(confidence * 100):.2f}%, {class_names[class_id]} | {int(area_percentage * 100):.2f}%"
                cv2.putText(
                    img= frame, 
                    text= label, 
                    org= (int(x_min), int(y_min) - 10), 
                    fontFace= cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale= 0.5, 
                    color= color, 
                    thickness= 1
                    )
            else:
                color=(255, 0, 0)
                cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color=color, thickness=2)
                label = f"{int(confidence * 100)}%, {class_names[class_id]}"
                cv2.putText(
                    img= frame, 
                    text= label, 
                    org= (int(x_min), int(y_min) - 10), 
                    fontFace= cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale= 0.5, 
                    color= color, 
                    thickness= 1
                    )

        return frame
    
    def analyze_video(self, inference_path: str, points: list):
        
        cap = cv2.VideoCapture(inference_path)
        if not cap.isOpened():
            raise ValueError("Unable to open video source:", inference_path)
        

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            cv2.polylines(frame, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=2)

            frame = self.process_frame(frame, points)

            # Display the frame with annotations
            cv2.imshow("Safezone Analysis", frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
   
        cap.release()
        cv2.destroyAllWindows()

    def analyze_video_streamlit(self, inference_path: str, points: list, frame_placeholder):
        """
        Analyze video and stream the results to Streamlit.
        
        :param inference_path: Path to the video file.
        :param points: List of points defining the safezone polygon.
        :param frame_placeholder: Streamlit placeholder for displaying frames.
        """
        cap = cv2.VideoCapture(inference_path)
        if not cap.isOpened():
            raise ValueError("Unable to open video source:", inference_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Draw the safezone polygon
            pts = np.array(points, dtype=np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

            # Overlay the coordinates of each point
            for i, (x, y) in enumerate(points):
                cv2.circle(frame, (int(x), int(y)), radius=5, color=(0, 0, 255), thickness=-1)  # Draw a small circle
                cv2.putText(
                    frame,
                    f"({int(x)}, {int(y)})",
                    (int(x) + 10, int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )

            # Perform object detection
            result = self.predict_object_detection(frame)

            bounding_boxes = result[0].boxes.xyxy.cpu().numpy()
            confidences = result[0].boxes.conf.cpu().numpy()
            classes = result[0].boxes.cls.cpu().numpy()
            class_names = result[0].names

            for i, box in enumerate(bounding_boxes):
                x_min, y_min, x_max, y_max = box[:4]
                confidence = confidences[i]
                class_id = int(classes[i])

                if class_names[class_id] == 'toddler':
                    # Check how much of the bounding box is inside the safezone
                    bbox = Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])
                    safezone = Polygon(points)
                    intersection = safezone.intersection(bbox)
                    intersection_area = intersection.area
                    bbox_area = bbox.area

                    if intersection_area == bbox_area:
                        area_percentage = 1
                    else:
                        area_percentage = intersection_area / bbox_area

                    if area_percentage >= 0.5:
                        color = (0, 255, 0)
                    else:
                        color = (0, 0, 255)

                    cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color=color, thickness=2)

                    label = f"{int(confidence * 100):.2f}%, {class_names[class_id]} | {int(area_percentage * 100):.2f}%"
                    cv2.putText(
                        img=frame,
                        text=label,
                        org=(int(x_min), int(y_min) - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=color,
                        thickness=1
                    )
                else:
                    color = (255, 0, 0)
                    cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color=color, thickness=2)
                    label = f"{int(confidence * 100)}%, {class_names[class_id]}"
                    cv2.putText(
                        img=frame,
                        text=label,
                        org=(int(x_min), int(y_min) - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=color,
                        thickness=1
                    )

            # Convert the frame to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display the frame in Streamlit
            frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

        cap.release()
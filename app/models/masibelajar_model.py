from typing import List
import cv2
import numpy as np
from shapely import Polygon
from ultralytics import YOLO

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)

class MasiBelajarModel:
    def __init__(self, od_weight : str, pose_weight : str):
        self.od_weight = od_weight
        self.pose_weight = pose_weight

        self.od_model = YOLO(self.od_weight)
        self.pose_model = YOLO(self.pose_weight)

        self.__load_icons()

    def __load_icons(self):
        self.icons = {
            "falling": cv2.imread("app/assets/falling.png"),
            "toddler": cv2.imread("app/assets/child_care.png"),
            "adult": cv2.imread("app/assets/person.png"),
        }

    def icon(self, label: str, size: int = 24):
        mapper = {
            "falling": ["falling", "fall"],
            "toddler": ["toddler", "child"],
            "adult": ["adult", "non-toddler", "non toddler"],
        }

        selected_key = None
        for key, values in mapper.items():
            if label in values:
                selected_key = key
                break
        else:
            raise ValueError(f"Label '{label}' not found in icon mapper.")
        
        resize = cv2.resize(self.icons[selected_key], (size, size))
        
        return resize

    def calculate_safezone_intersection_percentage(self, bbox: Polygon, safezone: Polygon) -> float:
        intersection = safezone.intersection(bbox)
        intersection_area = intersection.area
        bbox_area = bbox.area

        if intersection_area == bbox_area:
            return 1
        else:
            return intersection_area / bbox_area
        
    
    def check_falling_pose(self, keypoints):
        """
        Check if a person is in a falling pose based on keypoints.
    
        Args:
            keypoints (list): List of keypoints (x, y, confidence) for a person.
    
        Returns:
            bool: True if the person is in a falling pose, False otherwise.
        """
        # Convert keypoints to a NumPy array for easier processing
        keypoints = np.array(keypoints)
    
        # Validate keypoints with confidence > 0.5
        keypoints = keypoints[keypoints[:, 2] > 0.5]

        score = {
            "upper_body" : np.average(keypoints[:6, 1]),
            "middle_body" : np.average(keypoints[6:10, 1]),
            "lower_body" : np.average(keypoints[10:, 1]),
        }

        return score["upper_body"] > score["lower_body"]
    
    def match_keypoints_to_bboxes(self, keypoints, bboxes):
        """
        Match keypoints to bounding boxes.
    
        Args:
            keypoints (np.ndarray): Array of keypoints (N x 3), where each row is (x, y, confidence).
            bboxes (np.ndarray): Array of bounding boxes (M x 4), where each row is (x_min, y_min, x_max, y_max).
    
        Returns:
            dict: A dictionary mapping each bounding box index to a list of keypoints.
        """
        matched_keypoints = {i: [] for i in range(len(bboxes))}
    
        # Iterate over each set of keypoints (e.g., for each person detected)
        for person_keypoints in keypoints:
            for x, y, confidence in person_keypoints:
                # Match each keypoint to a bounding box
                for i, bbox in enumerate(bboxes):
                    x_min, y_min, x_max, y_max = bbox
                    if x_min <= x <= x_max and y_min <= y <= y_max:
                        matched_keypoints[i].append((x, y, confidence))
    
        return matched_keypoints


    def analyze_frame(self, 
                  inference_path: str, 
                  safezone_points: List, 
                  target_class: List[str] = ['toddler', 'non-toddler'],
                  preview: bool = False,
                  stream: bool = False,
                  verbose: bool = False):
        """Analyze a video frame by frame and check if a person is falling or out of the safezone.
        This function uses the object detection model to detect people and the pose estimation model to check their poses.
        It also checks if the detected person is out of the safezone defined by the given points.
        The function yields a dictionary with the results and the frame with overlays if preview is True.

        Args:
            inference_path (str): Inference path to the video or image.
            safezone_points (List): (x, y) points of the safezone polygon.
            target_class (List[str], optional): Target class which want to be supervised. Defaults to ['toddler', 'non-toddler'].
            preview (bool, optional): Plot the inference or prediction result. Defaults to False.
            stream (bool, optional): Read inference path as generator. Defaults to False.
            verbose (bool, optional): Verbose message output. Defaults to False.

        Yields:
            _type_: A tuple containing a dictionary with the results and the frame with overlays if preview is True.
        """
        safezone_points = np.array(safezone_points)

        for od_result in self.od_model.predict(inference_path, stream=stream, verbose=verbose):
            # Object detection results
            bboxes = od_result.boxes.xyxy.cpu().numpy()
            confidences = od_result.boxes.conf.cpu().numpy()
            pred_classes = od_result.boxes.cls.cpu().numpy()
            labels = od_result.names

            # Run pose estimation on the full frame
            pose_result = self.pose_model.predict(od_result.orig_img, verbose=verbose)[0]
            keypoints_results = pose_result.keypoints.data.cpu().numpy()

            # Match keypoints to bounding boxes
            matched_keypoints = self.match_keypoints_to_bboxes(keypoints_results, bboxes)

            is_person_fall = False
            is_person_out_of_safezone = False

            if preview:
                frame = od_result.plot(
                    line_width=1,
                    labels=False,
                )
            else:
                frame = None

            for i, bbox in enumerate(bboxes):
                x_min, y_min, x_max, y_max = bbox[:4]
                confidence = confidences[i]

                if confidence < 0.25:
                    continue

                try:
                    label = labels[int(pred_classes[i])]
                except IndexError:
                    print(f"IndexError: {pred_classes[i]} not in labels")
                    continue

                if label in target_class:
                    safezone = Polygon(safezone_points)
                    target_bbox = Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])
                    intersection_percentage = self.calculate_safezone_intersection_percentage(target_bbox, safezone)

                    is_person_out_of_safezone = intersection_percentage < 0.5

                    # Check for falling pose
                    keypoints = matched_keypoints[i]
                    if keypoints:
                        is_person_fall = self.check_falling_pose(keypoints)

                    if preview:
                       

                        # Overlay the icon
                        icon_alpha = 0.8

                        sz_icon = self.icon(label)
                        sz_color = YELLOW if is_person_out_of_safezone else GREEN
                        sz_icon_tint = np.full_like(sz_icon, sz_color, dtype=np.uint8)
                        sz_icon_tint = cv2.addWeighted(sz_icon, 0.5, sz_icon_tint, 0.5, 0)

                        icon_height, icon_width = sz_icon.shape[:2]

                        y_start_pos = int(y_min)
                        y_end_pos = int(y_min) + icon_height

                        x_start_pos_1 = int(x_min)
                        x_end_pos_1 = int(x_min) + icon_width

                        x_start_pos_2 = int(x_max) - icon_width
                        x_end_pos_2 = int(x_max)

                        if is_person_fall:
                            ps_icon = self.icon("falling")

                            ps_icon_tint = np.full_like(ps_icon, RED, dtype=np.uint8)
                            ps_icon_tint = cv2.addWeighted(ps_icon, icon_alpha, ps_icon_tint, 0.5, 0)

                            frame[y_start_pos:y_end_pos, x_start_pos_2:x_end_pos_2] = cv2.addWeighted(
                                frame[y_start_pos:y_end_pos, x_start_pos_2:x_end_pos_2], icon_alpha, ps_icon_tint, 0.5, 0)

                        # Overlay the icon on the frame
                        frame[y_start_pos:y_end_pos, x_start_pos_1:x_end_pos_1] = cv2.addWeighted(
                            frame[y_start_pos:y_end_pos, x_start_pos_1:x_end_pos_1],icon_alpha, sz_icon_tint, 0.5, 0)

                        # Safezone percentage label
                        color = RED if is_person_out_of_safezone or is_person_fall else GREEN
                        label_text = f"{(intersection_percentage * 100):.2f}%"
                        cv2.putText(
                            img=frame,
                            text=label_text,
                            org=(int(x_min), int(y_min) - 10),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            color=color,
                            thickness=1
                        )

                        # Draw safezone polygon
                        cv2.polylines(frame, [safezone_points.astype(np.int32)], isClosed=True, color=color, thickness=2)

            yield ({
                "is_person_fall": is_person_fall,
                "is_person_out_of_safezone": is_person_out_of_safezone,
            }, frame)
                









        
from collections import defaultdict
import cv2
import numpy as np
from shapely import Polygon
from ultralytics import YOLO
from pathlib import Path
from typing import List, Union
from PIL import Image
from datetime import datetime

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)

class MasiBelajarModel:
    def __init__(self, od_weight : str, pose_weight : str, tracker: str):
        self.od_weight = od_weight
        self.pose_weight = pose_weight
        self.tracker_config = tracker

        self.od_model = YOLO(self.od_weight)
        self.pose_model = YOLO(self.pose_weight)

        self.__load_icons()
        self.__setup_tracker()
        


    def analyze_frame(self, 
                  inference_path: Union[str, Path, int, Image.Image, list, tuple, np.ndarray], 
                  safezone_points: List, 
                  target_class: List[str] = ['toddler', 'non-toddler'],
                  preview: bool = False,
                  stream: bool = False,
                  track: bool = False,
                  reference_vector_tracker: np.ndarray = None,
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
        if track and reference_vector_tracker is None:
            raise ValueError("Reference vector tracker is required when track is True.")

        safezone_points = np.array(safezone_points)

        for od_result in self.od_model.track(inference_path, stream=stream, verbose=verbose, persist=True):
            # Object detection results
            bboxes = od_result.boxes.xyxy.cpu().numpy()
            confidences = od_result.boxes.conf.cpu().numpy()
            pred_classes = od_result.boxes.cls.cpu().numpy()
            labels = od_result.names

            # Track
            if track:
                try:
                    track_ids = od_result.boxes.id.int().cpu().numpy()
                    track_boxes = od_result.boxes.xywh.cpu().numpy()
                except AttributeError:
                    track_ids = None
                    track_boxes = None

            # Run pose estimation on the full frame
            pose_result = self.pose_model.predict(od_result.orig_img, verbose=verbose)[0]
            keypoints_results = pose_result.keypoints.data.cpu().numpy()

            # Match keypoints to bounding boxes
            matched_keypoints = self.__match_keypoints_to_bboxes(keypoints_results, bboxes)

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
                    continue

                if label in target_class:
                    safezone = Polygon(safezone_points)
                    target_bbox = Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])
                    intersection_percentage = self.__calculate_safezone_intersection_percentage(target_bbox, safezone)

                    is_person_out_of_safezone = intersection_percentage < 0.5

                    # Check for falling pose
                    keypoints = matched_keypoints[i]
                    if keypoints:
                        is_person_fall = self.__check_falling_pose(keypoints)

                    # Tracker
                    if track and track_ids is not None and track_boxes is not None:
                        x, y, w, h = track_boxes[i]
                        self.__update_tracker(
                            id=int(track_ids[i]), 
                            position=(float(x), float(y)), 
                            width=w, 
                            height=h
                            )
                        
                        is_flow_in = self.__check_vector_direction(
                            id=int(track_ids[i]), 
                            reference_vector=reference_vector_tracker
                            )
                        
                        self.__update_flow_in(
                            id=int(track_ids[i]), 
                            value=is_flow_in
                            )


                    if preview:
                        self.__draw_visuals(
                            safezone_points=safezone_points, 
                            is_person_fall=is_person_fall, 
                            is_person_out_of_safezone=is_person_out_of_safezone, 
                            frame=frame, 
                            x_min=x_min, 
                            y_min=y_min, 
                            x_max=x_max, 
                            label=label, 
                            intersection_percentage=intersection_percentage,
                            track=track,
                            tracks_ids=track_ids if track else None,
                            tracks_boxes=track_boxes if track else None,
                            )

            # Count each label
            label_counts = {}
            for label in target_class:
                label_counts[label] = sum(label in labels[int(pred_classes[i])] for i in range(len(pred_classes)))

            yield ({
                "fall": is_person_fall,
                "out_of_safezone": is_person_out_of_safezone,
                "counts": label_counts,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }, frame)

    def __load_icons(self):
        self.icons = {
            "falling": cv2.imread("app/assets/falling.png"),
            "toddler": cv2.imread("app/assets/child_care.png"),
            "adult": cv2.imread("app/assets/person.png"),
        }

    def __icon(self, label: str, size: int = 24):
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
    
    def __setup_tracker(self):
        self.tracker_history = defaultdict(lambda: {
            "entry_time": None,
            "last_seen": None,
            "previous_position": None,
            "current_position": None,
            "width": None,
            "height": None,
            "flow_in": False,
            "history": []
        })

    def __update_tracker(self, id, position, width, height, max_distance=50, tolerance=0.2):
        if id in self.tracker_history:
            # Update existing track
            self.tracker_history[id]["previous_position"] = self.tracker_history[id]["current_position"]
            self.tracker_history[id]["current_position"] = position
            self.tracker_history[id]["width"] = width
            self.tracker_history[id]["height"] = height
            self.tracker_history[id]["last_seen"] = datetime.now()
        else:
            # Extract positions from tracker history
            data = [
                (track_data["current_position"], track_data["width"], track_data["height"], track_id)
                for track_id, track_data in self.tracker_history.items()
                if track_data["current_position"] is not None
            ]
            
            # Unpack the data into separate lists
            positions, widths, heights, track_ids = zip(*data) if data else ([], [], [], [])
    
            if positions:
                # Use NumPy for nearest-neighbor search
                positions = np.array(positions)
                distances = np.linalg.norm(positions - np.array(position), axis=1)
    
                valid_indices = np.where(
                    (distances <= max_distance) &
                    (np.abs(np.array(widths) - width) / width <= tolerance) &
                    (np.abs(np.array(heights) - height) / height <= tolerance)
                )[0]
    
                if len(valid_indices) > 0:
                    nearest_index = valid_indices[np.argmin(distances[valid_indices])]
                    original_id = track_ids[nearest_index]
    
                    self.tracker_history[id] = {
                        "entry_time": self.tracker_history[original_id]["entry_time"],
                        "last_seen": datetime.now(),
                        "previous_position": self.tracker_history[original_id]["current_position"],
                        "current_position": position,
                        "width": width,
                        "height": height,
                        "flow_in":  self.tracker_history[original_id]["flow_in"],
                        "history": self.tracker_history[original_id]["history"]
                    }

                    # Remove the old track
                    del self.tracker_history[original_id]
                    return
    
            # If no match is found, create a new track
            self.tracker_history[id] = {
                "entry_time": datetime.now(),
                "last_seen": datetime.now(),
                "previous_position": None,
                "current_position": position,
                "width": width,
                "height": height,
                "flow_in": False,
                "history": []
            }

    def __check_vector_direction(self, id, reference_vector):
        if id in self.tracker_history:
            current_position = self.tracker_history[id]["current_position"]
            previous_position = self.tracker_history[id]["previous_position"]

            if current_position is not None and previous_position is not None:
                vector = np.array(current_position) - np.array(previous_position)
                distance = np.linalg.norm(vector)

                if distance > 0:
                    direction_vector = vector / distance

                    dot_product = np.dot(direction_vector, reference_vector)

                    return dot_product > 0
                elif distance == 0:
                    return self.tracker_history[id]["flow_in"]

        return False
    
    def __update_flow_in(self, id, value: bool):
        if id in self.tracker_history:
            self.tracker_history[id]["flow_in"] = value
        else:
            raise ValueError(f"ID {id} not found in tracker history.")

    
    def __calculate_safezone_intersection_percentage(self, bbox: Polygon, safezone: Polygon) -> float:
        intersection = safezone.intersection(bbox)
        intersection_area = intersection.area
        bbox_area = bbox.area

        if intersection_area == bbox_area:
            return 1
        else:
            return intersection_area / bbox_area
        
    
    def __check_falling_pose(self, keypoints) -> bool:
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

        return bool(score["upper_body"] > score["lower_body"])

    def __match_keypoints_to_bboxes(self, keypoints, bboxes):
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

    def __draw_visuals(self, safezone_points: np.ndarray, is_person_fall: bool, is_person_out_of_safezone:bool, frame:np.ndarray, x_min, y_min, x_max, label, intersection_percentage: float, track: bool, tracks_ids: np.ndarray, tracks_boxes: np.ndarray) -> None:
        if track and (tracks_ids is None or tracks_boxes is None):
            raise ValueError("Track IDs and boxes are required when track is True.")
        
        # Overlay the icon
        icon_alpha = 0.8

        sz_icon = self.__icon(label)
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
            ps_icon = self.__icon("falling")

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

        # Draw track
        if track:
            try:
                for box, track_id in zip(tracks_boxes, tracks_ids):
                    x, y, _, _ = box
                    tracker = self.tracker_history[track_id]
                    if tracker["flow_in"]:
                        color = GREEN
                    else:
                        color = RED
                    tracker["history"].append((float(x), float(y)))
                    if len(tracker["history"]) > 30:
                        tracker["history"].pop(0)
                    points = np.hstack(tracker["history"]).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], isClosed=False, color=color, thickness=5)

                    # Show track ID
                    cv2.putText(
                        img=frame,
                        text=str(track_id),
                        org=(int(x), int(y) - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=color,
                        thickness=1
                    )
            except Exception as e:
                print(f"Error in drawing track: {e}")
                









        
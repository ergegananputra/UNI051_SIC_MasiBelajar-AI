from collections import defaultdict
import os
from weakref import ref
import cv2
import numpy as np
from ultralytics import YOLO
from app.models import SafezoneModel
from app.models.key_points.pose_model import PoseModel
from app.models import MasiBelajarModel
from test.utils.decorators import show_func_name

@show_func_name
def test_object_detection(model : SafezoneModel, image_path: str):
    model.predict_object_detection(image_path, show=True)

@show_func_name
def test_analyze_od_video(model : SafezoneModel, image_path: str):
    safezone = [
        (277, 142),
        (299, 149),
        (454, 140),
        (449, 219),
        (301, 244),
        (277, 225)
    ]
    model.analyze_video(image_path, safezone)


@show_func_name
def test_pose(model : PoseModel, image_path: str):
    model.inference(image_path, show=True)

@show_func_name
def test_analyze_pose_video(model : PoseModel, image_path: str):
    model.analyze_video(image_path)

@show_func_name
def test_masibelajar_model(image_path: str, safezone: list, save: bool = False):
    model = MasiBelajarModel(
        od_weight='app/models/object_detection/config/best.pt',
        pose_weight='app/models/key_points/config/yolo11m-pose.pt',
        tracker='app/models/tracker/tracker.yaml',
    )

    if save:
        output_dir = "storages/output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        _counter = 0

    for summary, frame in model.analyze_frame(
        inference_path=image_path, 
        preview=True, 
        safezone_points=safezone,
        stream=True,
        verbose=False,
        track=True,
        time_threshold=5
        ):

        cv2.imshow("Frame", frame)

        if save:
            _counter += 1
            cv2.imwrite(os.path.join(output_dir, f"{_counter}.jpg"), frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


@show_func_name
def test_masibelajar_model_2(image_path: str, safezone: list):
    model = MasiBelajarModel(
        od_weight='app/models/object_detection/config/best.pt',
        pose_weight='app/models/key_points/config/yolo11m-pose.pt',
        tracker='app/models/tracker/tracker.yaml',
    )

    point_1 = (370, 156)
    point_2 = (350, 192)

    vector = np.array(point_2) - np.array(point_1)
    distance = np.linalg.norm(vector)
    direction_vector = vector / distance

    reference_vector_tracker = direction_vector

    # cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture(image_path)
    cap = cv2.VideoCapture("http://192.168.122.82:81/stream")

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        for summary, frame in model.analyze_frame(
            inference_path=img, 
            preview=True, 
            safezone_points=safezone,
            stream=True,
            verbose=False,
            track=True,
            ):

            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

@show_func_name
def test_tracking(img: str):
    model = YOLO('app/models/object_detection/config/best.pt')
    track_history = defaultdict(lambda: [])
    for result in model.track(
        img, 
        stream=True, 
        verbose=True,
        tracker='app/models/tracker/tracker.yaml',
        # tracker='botsort.yaml',
        persist=True,
        ):
        frame = result.plot()

        try:
            boxes = result.boxes.xywh.cpu()
            track_ids = result.boxes.id.int().cpu().tolist()


            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))
                if len(track) > 30:
                    track.pop(0)

                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
        except Exception as e:
            print(e)

        cv2.imshow("Frame Tracking", frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    # safezone = [[277, 142],[299, 149],[454, 140],[449, 219],[301, 244],[277, 225]]
    # safezone = [[442,30], [496, 33], [488,201], [437, 205]]
    # safezone = [[319, 49], [478, 114], [479, 328], [328, 407]] # Scena 1
    # safezone = [[559, 138], [686, 172], [650, 508], [521, 534]] # Scena 2
    safezone = [[696, 210], [1200, 130], [1166, 716], [1009, 718], [705, 567]] # Scena 3

    # image_path = 'test/data/Fall.mp4'
    # image_path = 'test/data/TikTokToddler.mp4'
    # image_path = 'https://www.youtube.com/live/yNKvkPJl-tg?feature=shared'
    # image_path = '/mnt/d/UGM/EDU/SamsungInnovationCampus/Dataset/Scena1.mp4'
    # image_path = '/mnt/d/UGM/EDU/SamsungInnovationCampus/Dataset/Scena2.mp4'
    # image_path = 'http://192.168.122.82:81/stream'
    image_path = '/mnt/d/UGM/EDU/SamsungInnovationCampus/Dataset/Stream.mp4'


    safeZoneModel : SafezoneModel = SafezoneModel()
    poseModel : PoseModel = PoseModel()

    # test_analyze_od_video(safeZoneModel, image_path)
    # test_object_detection(safeZoneModel, image_path)
    # test_pose(poseModel, image_path)
    # test_analyze_pose_video(poseModel, image_path)


    test_masibelajar_model(image_path, safezone, True)
    # test_masibelajar_model_2(image_path, safezone)

    # test_tracking(image_path)


    # poseModel.stream_webcam()




    

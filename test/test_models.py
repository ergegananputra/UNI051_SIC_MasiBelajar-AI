import cv2
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
def test_masibelajar_model(image_path: str, safezone: list):
    model = MasiBelajarModel(
        od_weight='app/models/object_detection/config/weight.pt',
        pose_weight='app/models/key_points/config/yolo11m-pose.pt'
    )

    for _, frame in model.analyze_frame(
        inference_path=image_path, 
        preview=True, 
        safezone_points=safezone,
        stream=True,
        verbose=False,
        ):
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    safezone = [
        (277, 142),
        (299, 149),
        (454, 140),
        (449, 219),
        (301, 244),
        (277, 225)
    ]

    # image_path = 'test/data/TikTokToddler.mp4'
    image_path = 'test/data/Fall.mp4'

    safeZoneModel : SafezoneModel = SafezoneModel()
    poseModel : PoseModel = PoseModel()

    # test_analyze_od_video(safeZoneModel, image_path)
    # test_object_detection(safeZoneModel, image_path)
    # test_pose(poseModel, image_path)
    # test_analyze_pose_video(poseModel, image_path)


    test_masibelajar_model(image_path, safezone)


    # poseModel.stream_webcam()




    

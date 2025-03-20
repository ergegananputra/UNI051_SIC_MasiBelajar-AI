from app.models import SafezoneModel
from app.models.key_points.pose_model import PoseModel
from test.utils.decorators import show_func_name

@show_func_name
def test_object_detection(model : SafezoneModel):
    model.predict_object_detection('test/data/TikTokToddler.mp4', show=True)

@show_func_name
def test_analyze_video(model : SafezoneModel):
    safezone = [
        (277, 142),
        (299, 149),
        (454, 140),
        (449, 219),
        (301, 244),
        (277, 225)
    ]
    model.analyze_video('test/data/TikTokToddler.mp4', safezone)


@show_func_name
def test_pose(model : PoseModel):
    model.inference('test/data/TikTokToddler.mp4', show=True)


if __name__ == '__main__':
    safeZoneModel : SafezoneModel = SafezoneModel()
    poseModel : PoseModel = PoseModel()

    # test_object_detection(safeZoneModel)
    # test_pose(poseModel)


    poseModel.stream_webcam()




    

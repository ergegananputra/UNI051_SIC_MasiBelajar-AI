import logging
import tempfile
import cv2
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import StreamingResponse
from app.models.masibelajar_model import MasiBelajarModel
from .configs import ENV

API_ROUTER = APIRouter()
logger = logging.getLogger(__name__)
model = MasiBelajarModel(
    od_weight=ENV.AI_MODEL_OBJECTS_DETECTION_WEIGHT_PATH,
    pose_weight=ENV.AI_MODEL_KEYPOINTS_WEIGHT_PATH,
    tracker=ENV.AI_MODEL_TRACKER_PATH,
)


@API_ROUTER.post("/stream")
def camera_stream(
    stream_url: str,
    points: list, 
    target_class: list = ["toddler", "non-toddler"],
    ):
    """
    Start the camera stream and return a streaming response with given video source.
    """

    def stream():
        for result, frame in model.analyze_frame(
            inference_path=stream_url,
            safezone_points=points,
            target_class=target_class,
            preview=True,
            stream=True,
            verbose=True,
            track=True,
        ):

            _, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
            yield (b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        
    return StreamingResponse(stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@API_ROUTER.post("/upload-video")
def upload_video_stream(
    file: UploadFile = File(...),
    points: list = [[100, 100], [200, 100], [200, 200], [100, 200]],  # Default safezone points
    target_class: list = ["toddler", "non-toddler"],  # Default target classes
):
    """
    Accept an uploaded video file and stream the processed video back to the client.
    """

    def stream():
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(file.file.read())
            temp_video_path = temp_video.name

        # Process the video using the model
        for result, frame in model.analyze_frame(
            inference_path=temp_video_path,
            safezone_points=points,
            target_class=target_class,
            preview=True,
            stream=True,
            verbose=True,
            track=True,
        ):
            # Encode the frame as JPEG
            _, buffer = cv2.imencode(".jpg", frame)
            frame_bytes = buffer.tobytes()

            # Yield the frame as part of the HTTP response
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    return StreamingResponse(stream(), media_type="multipart/x-mixed-replace; boundary=frame")
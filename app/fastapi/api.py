import logging
import cv2
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.models.masibelajar_model import MasiBelajarModel
from configs import ENV

API_ROUTER = APIRouter()
logger = logging.getLogger(__name__)
model = MasiBelajarModel(
    od_weight=ENV.AI_MODEL_OBJECTS_DETECTION_WEIGHT_PATH,
    pose_weight=ENV.AI_MODEL_KEYPOINTS_WEIGHT_PATH,
    tracker=ENV.AI_MODEL_TRACKER_PATH,
)

def stream_service():
    # TODO: Implement the stream_service function
    raise NotImplementedError("stream_service function is not implemented yet.")

@API_ROUTER.post("/stream")
def camera_stream():
    return StreamingResponse(stream_service(), media_type="multipart/x-mixed-replace; boundary=frame")
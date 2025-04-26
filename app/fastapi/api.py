import logging
import tempfile
from typing import List, Optional
import cv2
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import StreamingResponse
from idna import encode
from pydantic import BaseModel
from app.models.masibelajar_model import MasiBelajarModel
from .configs import ENV

API_ROUTER = APIRouter()
logger = logging.getLogger(__name__)

model = MasiBelajarModel(
    od_weight=ENV.AI_MODEL_OBJECTS_DETECTION_WEIGHT_PATH,
    pose_weight=ENV.AI_MODEL_KEYPOINTS_WEIGHT_PATH,
    tracker=ENV.AI_MODEL_TRACKER_PATH,
)


for results, frame in model.analyze_frame(
            inference_path="storages/sample/Stream.mp4",
            safezone_points= [[696, 210], [1200, 130], [1166, 716], [1009, 718], [705, 567]],
            time_threshold=50,
            id="test",
            preview=True,
            stream=True,
            track=True,
            verbose=False,
        ): 
    print(results)


@API_ROUTER.get("")
def home():
    return {"message": "Welcome to Lokari API"}


class StreamRequest(BaseModel):
    id: str
    points: List[List[int]]
    url: Optional[str] = None
    time_threshold: int = 3600
    developer_demo_preview: bool = False
    developer_auto_break_threshold: int = 500

@API_ROUTER.post("/stream")
def camera_stream(request: StreamRequest):
    id = request.id
    points = request.points
    url = request.url
    time_threshold = request.time_threshold
    developer_demo_preview = request.developer_demo_preview
    developer_auto_break_threshold = request.developer_auto_break_threshold


    if url is None and not developer_demo_preview:
        return {"message": "Please provide a URL for the camera stream."}
    
    if url is None and developer_demo_preview:
        url = "storages/sample/Stream.mp4"
        points = [[696, 210], [1200, 130], [1166, 716], [1009, 718], [705, 567]]

    def stream_video(url: str, points: List[List[int]], time_threshold: int, id: str, **kwargs):
        threshold = kwargs["developer_auto_break_threshold"]

        for results, frame in model.analyze_frame(
                inference_path=url,
                safezone_points=points,
                time_threshold=time_threshold,
                id=id,
                preview=True,
                stream=True,
                track=True,
                verbose=False,
            ): 

            if frame is not None:
                # Encode the frame as JPEG
                _, encoded_frame = cv2.imencode('.jpg', frame)

                cv2.imshow("Frame", frame)

                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + encoded_frame.tobytes() + b'\r\n')
            else:
                threshold -= 1
                if threshold <= 0:
                    break
                continue
            
    
    
    return StreamingResponse(
        stream_video(url, points, time_threshold, id, developer_auto_break_threshold=developer_auto_break_threshold),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
    

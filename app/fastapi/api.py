import asyncio
import base64
import json
import logging
import tempfile
from typing import List, Optional
import cv2
from fastapi import APIRouter, BackgroundTasks, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from idna import encode
from pydantic import BaseModel
from app.fastapi.stream_manager import StreamManager
from app.models.masibelajar_model import MasiBelajarModel
from .configs import ENV

API_ROUTER = APIRouter()
logger = logging.getLogger(__name__)

def get_model() -> MasiBelajarModel:
    return MasiBelajarModel(
        od_weight=ENV.AI_MODEL_OBJECTS_DETECTION_WEIGHT_PATH,
        pose_weight=ENV.AI_MODEL_KEYPOINTS_WEIGHT_PATH,
        tracker=ENV.AI_MODEL_TRACKER_PATH,
    )

model = get_model()

stream_manager = StreamManager()


@API_ROUTER.get("")
def home():
    return {"message": "Welcome to Lokari API"}


async def stream_video_via_websocket(websocket: WebSocket, video_path: str):
    await websocket.accept()
    logger.error("WebSocket connection established")

    try:
        id = "dummy_stream"
        points = [[696, 210], [1200, 130], [1166, 716], [1009, 718], [705, 567]]
        url = video_path
        time_threshold = 60

        logger.error("Starting analyze_frame loop")

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
            
            json_data = {
                "message": "Prediction task running",
                "data": {
                    "results": results,
                    "frame": None
                }
            }

            if frame is not None:
                _, buffer = cv2.imencode('.jpeg', frame)
                frame_data = base64.b64encode(buffer).decode('utf-8')
                json_data["data"]["frame"] = frame_data

            await websocket.send_text(json.dumps(json_data))

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error during WebSocket streaming: {e}")
    finally:
        logger.info("Closing WebSocket connection")
        try:
            await websocket.close()
        except Exception as close_error:
            logger.error(f"Error while closing WebSocket: {close_error}")

@API_ROUTER.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    video_path = "storages/sample/Stream.mp4"
    await stream_video_via_websocket(websocket, video_path)



@API_ROUTER.websocket("/main-con")
async def ws_main_con(websocket: WebSocket):
    """WebSocket Main Connection Handler

    Configs:
    - url : str
    - points : List[List[int]]
    - time_threshold : int = 3600
    - id : str 
    - preview : bool = False
    - track : bool = False
    """
    await websocket.accept()

    shared_prediction_task = None

    async def run_prediction_task(config: dict):
        try:
            # Validate the configuration
            if not config.get("url"):
                raise ValueError("Missing 'url' in configuration")
            if not config.get("points"):
                raise ValueError("Missing 'points' in configuration")
            if not isinstance(config.get("points"), list):
                raise ValueError("'points' must be a list of coordinates")
            if not config.get("id"):
                raise ValueError("Missing 'id' in configuration")

            if config["reset_prediction_task"] == True:
                global model
                model = get_model()

            # Start the stream and get the frame queue
            frame_queue = await stream_manager.start_stream(config["url"])
            client_queue = asyncio.Queue()
            stream_manager.streams[config["url"]]["clients"].add(client_queue)

            try:
                while True:
                    # Get the next frame from the queue
                    frame = await client_queue.get()

                    # Process the frame with YOLO
                    for results, processed_frame in model.analyze_frame(
                        inference_path=config["url"],
                        safezone_points=config["points"],
                        time_threshold=config["time_threshold"],
                        id=config["id"],
                        preview=config["preview"],
                        stream=True,
                        track=config["track"],
                        verbose=False,
                    ):
                        json_data = {
                            "message": "Prediction task running",
                            "data": {
                                "results": results,
                                "frame": None
                            }
                        }

                        if processed_frame is not None:
                            _, buffer = cv2.imencode('.jpeg', processed_frame)
                            frame_data = base64.b64encode(buffer).decode('utf-8')
                            json_data["data"]["frame"] = frame_data

                        await websocket.send_text(json.dumps(json_data))

            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for camera: {config['url']}")
            finally:
                # Remove the client and stop the stream if no clients are left
                stream_manager.streams[config["url"]]["clients"].remove(client_queue)
                await stream_manager.stop_stream(config["url"])

        except ValueError as ve:
            logger.error(f"Configuration error: {ve}")
            await websocket.send_text(json.dumps({"error": str(ve)}))
        except asyncio.CancelledError:
            logger.info("Prediction task was canceled")
        except Exception as e:
            logger.error(f"Error during prediction task: {e}")

    try:
        while True:
            # Wait for a new configuration from the client
            data = await websocket.receive_text()

            try:
                config = json.loads(data)
                config.setdefault("time_threshold", 3600)
                config.setdefault("reset_prediction_task", True)

                # Cancel the current prediction task if it exists
                if shared_prediction_task and not shared_prediction_task.done():
                    logger.info("Stopping previous prediction task")
                    shared_prediction_task.cancel()
                    await shared_prediction_task  # Wait for the task to finish

                # Start a new prediction task with the updated configuration
                shared_prediction_task = asyncio.create_task(run_prediction_task(config))

            except json.JSONDecodeError:
                logger.error("Invalid JSON received")
                await websocket.send_text(json.dumps({"error": "Invalid JSON format"}))
            except Exception as e:
                logger.error(f"Error processing data: {e}")
                await websocket.send_text(json.dumps({"error": str(e)}))

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error during WebSocket communication: {e}")
        logger.info("Closing WebSocket connection")
        await websocket.close()
    finally:
        logger.info("WebSocket connection closed")
        if shared_prediction_task and not shared_prediction_task.done():
            shared_prediction_task.cancel()
            await shared_prediction_task
            logger.info("Prediction task canceled")

        try:
            await websocket.close()
        except Exception as close_error:
            logger.error(f"Error while closing WebSocket: {close_error}")

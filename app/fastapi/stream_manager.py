import asyncio

import cv2
import logging

logger = logging.getLogger(__name__)

class StreamManager:
    def __init__(self):
        self.streams = {}  # Dictionary to manage active streams

    async def start_stream(self, url: str):
        """Start a new stream for the given URL."""
        if url not in self.streams:
            self.streams[url] = {
                "clients": set(),
                "task": asyncio.create_task(self._stream_frames(url)),
                "queue": asyncio.Queue(),
            }
        return self.streams[url]["queue"]

    async def stop_stream(self, url: str):
        """Stop the stream if no clients are connected."""
        if url in self.streams and not self.streams[url]["clients"]:
            self.streams[url]["task"].cancel()
            del self.streams[url]

    async def _stream_frames(self, url: str):
        """Fetch frames from the camera and distribute them to the queue."""
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            logger.error(f"Failed to open camera stream: {url}")
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error(f"Failed to read frame from camera: {url}")
                    break

                # Put the frame in the queue for clients
                for client_queue in self.streams[url]["clients"]:
                    await client_queue.put(frame)

                await asyncio.sleep(0.03)  # Simulate ~30 FPS
        except asyncio.CancelledError:
            logger.info(f"Stream for {url} canceled")
        finally:
            cap.release()
            logger.info(f"Camera stream closed: {url}")
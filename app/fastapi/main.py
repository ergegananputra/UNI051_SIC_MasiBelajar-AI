import os
import logging
from datetime import datetime
from fastapi import FastAPI

from .configs import ENV
from app.fastapi import API_ROUTER 

os.makedirs("storages", exist_ok=True)
os.makedirs("storages/logs", exist_ok=True)

month, year = datetime.now().month, datetime.now().year

logging.basicConfig(
    filename="storages/logs/fastapi_{}_{}.log".format(month, year),
    level=logging.ERROR,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

# Create directories if they do not exist



app = FastAPI(
    title=ENV.APP_NAME,
    openapi_url="/o/documentation/openapi.json",
    version="0.1.0",
)

app.include_router(API_ROUTER, prefix="/v1", tags=["v1"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app=app, host=ENV.APP_HOST, port=ENV.APP_PORT, reload=True)


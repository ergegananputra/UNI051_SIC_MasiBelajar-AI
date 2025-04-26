from pydantic_settings import BaseSettings, SettingsConfigDict

class ConfigEnv(BaseSettings):
    APP_NAME: str
    APP_HOST: str 
    APP_PORT: int

    AI_MODEL_KEYPOINTS_WEIGHT_PATH: str
    AI_MODEL_OBJECTS_DETECTION_WEIGHT_PATH: str
    AI_MODEL_TRACKER_PATH: str

    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_file='.env'
    )

ENV = ConfigEnv()
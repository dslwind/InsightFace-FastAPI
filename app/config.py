from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # InsightFace Configuration
    INSIGHTFACE_MODEL_NAME: str = "buffalo_l"
    ENABLE_CUDA: bool = False
    
    # API Default Parameters
    DEFAULT_THRESHOLD: float = 0.3
    DEFAULT_DETECTION_THRESHOLD: float = 0.6
    DEFAULT_LIMIT_FACES: int = 0
    DEFAULT_MIN_FACE_SIZE: int = 0
    DEFAULT_RETURN_FACE_DATA: bool = False
    DEFAULT_RETURN_LANDMARKS: bool = False
    DEFAULT_BEST_FACE_STRATEGY: str = "center"
    DEFAULT_INPUT_FORMAT: str = "auto"
    DEFAULT_COMPARE_ALL_FACES: bool = False

    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = "logs"
    LOG_FILENAME: str = "insightface_api.log"
    LOG_MAX_BYTES: int = 10 * 1024 * 1024  # 10 MB
    LOG_BACKUP_COUNT: int = 5

    class Config:

        env_file = ".env"

settings = Settings()

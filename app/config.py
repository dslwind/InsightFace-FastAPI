from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # InsightFace Configuration
    INSIGHTFACE_MODEL_NAME: str = "buffalo_l"
    INSIGHTFACE_ROOT: str = "models"
    ENABLE_CUDA: bool = False
    ENABLE_TENSORRT: bool = False

    # API Default Parameters
    DEFAULT_THRESHOLD: float = 0.3
    DEFAULT_DETECTION_THRESHOLD: float = 0.6
    DEFAULT_LIMIT_FACES: int = 0
    DEFAULT_MIN_FACE_SIZE: int = 0
    DEFAULT_BEST_FACE_STRATEGY: str = "center"
    DEFAULT_COMPARE_ALL_FACES: bool = False
    DEFAULT_ENABLE_ROTATION: bool = False

    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = "logs"
    LOG_FILENAME: str = "insightface_api.log"
    LOG_MAX_BYTES: int = 10 * 1024 * 1024  # 10 MB
    LOG_BACKUP_COUNT: int = 5
    LOG_ROTATION_TYPE: str = "time"  # "size" or "time"
    LOG_ROTATION_WHEN: str = "midnight"  # Rotates at midnight
    LOG_ROTATION_INTERVAL: int = 1
    LOG_COMPRESS: bool = True

    # Database Logging Configuration
    ENABLE_DB_LOGGING: bool = False
    DB_POSTGRES_URL: str = ""  # e.g., postgresql://user:password@localhost:5432/dbname
    DB_SQLITE_URL: str = "sqlite:///logs/app_logs.db"

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()

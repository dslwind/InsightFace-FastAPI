import datetime
import json
import os
from sqlalchemy import Column, Integer, String, DateTime, Text, Float, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.config import settings

Base = declarative_base()

class ApiRequestLog(Base):
    __tablename__ = "api_request_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.datetime.now)
    client_ip = Column(String(50))
    method = Column(String(10))
    path = Column(String(255))
    request_params = Column(Text)  # JSON string of query params and body
    response_status = Column(Integer)
    response_body = Column(Text)   # JSON string of response
    processing_time = Column(Float) # Seconds

class DatabaseLogger:
    def __init__(self):
        self.engine = None
        self.Session = None
        self._init_db()

    def _init_db(self):
        if not settings.ENABLE_DB_LOGGING:
            return

        # 1. Try PostgreSQL
        if settings.DB_POSTGRES_URL:
            try:
                self.engine = create_engine(settings.DB_POSTGRES_URL, connect_args={'connect_timeout': 5})
                with self.engine.connect() as conn:
                    pass
                Base.metadata.create_all(self.engine)
                self.Session = sessionmaker(bind=self.engine)
                print("Successfully connected to PostgreSQL for API logging.")
                return
            except Exception as e:
                print(f"Failed to connect to PostgreSQL: {e}. Falling back to SQLite.")

        # 2. Fallback to SQLite
        try:
            if settings.DB_SQLITE_URL.startswith("sqlite:///"):
                db_path = settings.DB_SQLITE_URL.replace("sqlite:///", "")
                db_dir = os.path.dirname(db_path)
                if db_dir and not os.path.exists(db_dir):
                    os.makedirs(db_dir)

            self.engine = create_engine(settings.DB_SQLITE_URL)
            Base.metadata.create_all(self.engine)
            self.Session = sessionmaker(bind=self.engine)
            print(f"Connected to SQLite for API logging at: {settings.DB_SQLITE_URL}")
        except Exception as e:
            print(f"Failed to connect to SQLite: {e}. API logging will be disabled.")
            self.Session = None

    def log_request(self, client_ip, method, path, params, status, response, duration):
        if not self.Session:
            return

        try:
            session = self.Session()
            log_entry = ApiRequestLog(
                client_ip=client_ip,
                method=method,
                path=path,
                request_params=json.dumps(params, ensure_ascii=False) if isinstance(params, (dict, list)) else str(params),
                response_status=status,
                response_body=json.dumps(response, ensure_ascii=False) if isinstance(response, (dict, list)) else str(response),
                processing_time=duration,
                timestamp=datetime.datetime.now()
            )
            session.add(log_entry)
            session.commit()
            session.close()
        except Exception as e:
            print(f"Failed to write API log to database: {e}")

# Global instance
db_logger = DatabaseLogger()

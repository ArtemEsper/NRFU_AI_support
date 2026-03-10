from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import settings

# For SQLite, we need to allow multi-thread access
connect_args = {"check_same_thread": False} if settings.sqlalchemy_database_uri.startswith("sqlite") else {}

engine = create_engine(settings.sqlalchemy_database_uri, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

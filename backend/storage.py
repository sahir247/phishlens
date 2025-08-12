from contextlib import contextmanager
from dataclasses import dataclass
from sqlalchemy import create_engine, Integer, Float, String
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Mapped, mapped_column
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "phishlens.db")
DB_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DB_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

class Base(DeclarativeBase):
    pass

class DetectionEvent(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    url: Mapped[str] = mapped_column(String(2048), nullable=False)
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    reasons_json: Mapped[str] = mapped_column(String, nullable=False, default="[]")
    ts: Mapped[float] = mapped_column(Float, nullable=False)

Base.metadata.create_all(bind=engine)

@dataclass
class DB:
    @staticmethod
    @contextmanager
    def session():
        s = SessionLocal()
        try:
            yield s
        finally:
            s.close()

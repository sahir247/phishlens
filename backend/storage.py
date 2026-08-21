from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from sqlalchemy import create_engine, Integer, Float, String, Text, func, desc
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Mapped, mapped_column
import os
import json
from typing import Dict, Any, List, Optional

from config import DB_URL, DB_PATH

engine = create_engine(DB_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


class DetectionEvent(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    url: Mapped[str] = mapped_column(String(2048), nullable=False)
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(32), nullable=False, default="SAFE")
    brand_target: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, default="")
    reasons_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    structured_reasons_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    category_scores_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    meta_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    ts: Mapped[float] = mapped_column(Float, nullable=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "url": self.url,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "brand_target": self.brand_target or "",
            "reasons": json.loads(self.reasons_json or "[]"),
            "structured_reasons": json.loads(self.structured_reasons_json or "[]"),
            "category_scores": json.loads(self.category_scores_json or "{}"),
            "meta": json.loads(self.meta_json or "{}"),
            "ts": self.ts,
        }


# Auto-create tables and patch columns if needed
Base.metadata.create_all(bind=engine)

def patch_sqlite_columns():
    """Ensure newly added columns exist if an old SQLite db was present."""
    with engine.connect() as conn:
        try:
            cursor = conn.connection.cursor()
            cursor.execute("PRAGMA table_info(events)")
            existing_cols = {row[1] for row in cursor.fetchall()}
            
            new_columns = [
                ("risk_level", "VARCHAR(32) DEFAULT 'SAFE'"),
                ("brand_target", "VARCHAR(64) DEFAULT ''"),
                ("structured_reasons_json", "TEXT DEFAULT '[]'"),
                ("category_scores_json", "TEXT DEFAULT '{}'"),
                ("meta_json", "TEXT DEFAULT '{}'"),
            ]
            for col_name, col_type in new_columns:
                if col_name not in existing_cols:
                    cursor.execute(f"ALTER TABLE events ADD COLUMN {col_name} {col_type}")
            conn.connection.commit()
        except Exception:
            pass

patch_sqlite_columns()


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

    @staticmethod
    def get_stats() -> Dict[str, Any]:
        with DB.session() as s:
            total_scans = s.query(func.count(DetectionEvent.id)).scalar() or 0
            dangerous_count = s.query(func.count(DetectionEvent.id)).filter(DetectionEvent.risk_score >= 0.80).scalar() or 0
            suspicious_count = s.query(func.count(DetectionEvent.id)).filter(DetectionEvent.risk_score >= 0.50, DetectionEvent.risk_score < 0.80).scalar() or 0
            safe_count = total_scans - (dangerous_count + suspicious_count)
            avg_risk = s.query(func.avg(DetectionEvent.risk_score)).scalar() or 0.0

            # Top detected brands
            brand_rows = s.query(
                DetectionEvent.brand_target,
                func.count(DetectionEvent.id)
            ).filter(
                DetectionEvent.brand_target != "",
                DetectionEvent.brand_target.isnot(None)
            ).group_by(DetectionEvent.brand_target).order_by(desc(func.count(DetectionEvent.id))).limit(5).all()

            top_brands = [{"brand": r[0], "count": r[1]} for r in brand_rows]

            # Recent 10 scans
            recent_events = s.query(DetectionEvent).order_by(DetectionEvent.ts.desc()).limit(10).all()

            return {
                "total_scans": total_scans,
                "dangerous_count": dangerous_count,
                "suspicious_count": suspicious_count,
                "safe_count": safe_count,
                "avg_risk_score": round(float(avg_risk), 3),
                "top_brands": top_brands,
                "recent_scans": [e.to_dict() for e in recent_events]
            }

    @staticmethod
    def get_trend(hours: int = 24) -> List[Dict[str, Any]]:
        """Return hourly scan counts for the past `hours` hours.

        Returns a list of dicts ordered oldest-first:
          [{"hour": "2026-08-21T14:00Z", "total": 12, "dangerous": 3,
            "suspicious": 5, "safe": 4}, ...]
        """
        now_utc = datetime.now(timezone.utc)
        cutoff_ts = (now_utc - timedelta(hours=hours)).timestamp()

        # Build list of hour boundaries (UTC, oldest first)
        buckets: List[Dict[str, Any]] = []
        for h in range(hours, 0, -1):
            bucket_start = now_utc - timedelta(hours=h)
            bucket_end   = now_utc - timedelta(hours=h - 1)
            buckets.append({
                "hour":      bucket_start.strftime("%Y-%m-%dT%H:00Z"),
                "hour_ts":   bucket_start.timestamp(),
                "end_ts":    bucket_end.timestamp(),
                "total":     0,
                "dangerous": 0,
                "suspicious": 0,
                "safe":      0,
            })

        with DB.session() as s:
            rows = (
                s.query(DetectionEvent.ts, DetectionEvent.risk_score)
                .filter(DetectionEvent.ts >= cutoff_ts)
                .all()
            )

        for row_ts, score in rows:
            for b in buckets:
                if b["hour_ts"] <= row_ts < b["end_ts"]:
                    b["total"] += 1
                    if score >= 0.80:
                        b["dangerous"] += 1
                    elif score >= 0.50:
                        b["suspicious"] += 1
                    else:
                        b["safe"] += 1
                    break

        # Strip internal timestamps before returning
        return [
            {k: v for k, v in b.items() if k not in ("hour_ts", "end_ts")}
            for b in buckets
        ]


from database import Base
from sqlalchemy import Column, Integer, Boolean, String, Float, Date, DateTime
from datetime import datetime

class ScanResult(Base):
    __tablename__ = 'scan_results'
    id = Column(Integer, primary_key=True)
    brand = Column(String(80), nullable=False)
    batch_no = Column(String(80), nullable=False, unique=True)
    date = Column(Date, nullable=False)
    confidence = Column(Float, nullable=False)
    is_authentic = Column(Boolean, nullable=False)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    image_url = Column(String(200), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

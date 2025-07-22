from sqlalchemy import Column, Integer, Float, String, Boolean, DateTime
from database import Base
from datetime import datetime

class Scan(Base):
    __tablename__ = "scans"
    id = Column(Integer, primary_key=True, index=True)
    is_counterfeit = Column(Boolean)
    confidence = Column(Float)
    message = Column(String)
    latitude = Column(Float)
    longitude = Column(Float)
    brand = Column(String)
    date = Column(DateTime, default=datetime.utcnow)

class ScanResult:
    pass

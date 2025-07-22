from pydantic import BaseModel, field_validator
from typing import Optional
from datetime import datetime

class AdminLogin(BaseModel):
    username: str
    password: str

class PredictRequest(BaseModel):
    brand: str = "County"
    latitude: Optional[str] = "Unknown"
    longitude: Optional[str] = "Unknown"

    @field_validator("latitude", "longitude")
    @classmethod
    def validate_coordinates(cls, v, info):
        if v != "Unknown":
            try:
                coord = float(v)
                if info.field_name == 'latitude' and not -90 <= coord <= 90:
                    raise ValueError("Invalid latitude")
                if info.field_name == 'longitude' and not -180 <= coord <= 180:
                    raise ValueError("Invalid longitude")
            except ValueError:
                raise ValueError(f"Invalid {info.field_name} format")
        return v

class PredictResponse(BaseModel):
    id: int
    is_authentic: bool
    brand: str
    batch_no: str
    date: str
    confidence: str
    latitude: str
    longitude: str
    image_url: Optional[str]
    message: str


class LocationResponse(BaseModel):
    id: int
    lat: float
    lng: float
    is_authentic: bool
    brand: str
    batch_no: str
    confidence: str
    date: str
    image_url: str

class LocationsResponse(BaseModel):
    locations: list[LocationResponse]

class ScanCreate(BaseModel):
    is_counterfeit: bool
    confidence: float
    message: str
    latitude: float
    longitude: float
    brand: Optional[str] = None
    date: Optional[datetime] = None

class ScanOut(ScanCreate):
    id: int
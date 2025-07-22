from fastapi import APIRouter, File, UploadFile, Form, Depends, Query
from fastapi.responses import FileResponse
from sqlalchemy import text
from sqlalchemy.orm import Session
from database import SessionLocal, get_db
from schemas import PredictRequest, PredictResponse, LocationsResponse, ScanOut
from repository.predict import process_prediction
from models import ScanResult, Scan
from logger_config import setup_logging
import os
from datetime import datetime
from typing import Optional
from PIL import Image
import numpy as np

router = APIRouter(prefix="", tags=["prediction"])
logger = setup_logging(__name__)

model = None  # Assume model is set globally or via dependency

@router.get("/health")
async def health(db: SessionLocal = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "database_connected": True,
        }
    except Exception as e:
        logger.error(f"DB health check failed: {e}")
        return {
            "status": "unhealthy",
            "model_loaded": model is not None,
            "database_connected": False,
        }
    
def predict_image(file: UploadFile, latitude: float, longitude: float):
    image = Image.open(file.file).convert("RGB")
    image = image.resize((224, 224))  # Adjust if your model expects a different size
    arr = np.array(image) / 255.0
    arr = np.expand_dims(arr, axis=0)  # Shape: (1, 224, 224, 3)

    # Dummy input: adjust shape as needed (e.g., (1, 2) for two dummy features)
    dummy_input = np.zeros((1, 256), dtype=np.float32)
    location_input = np.array([[latitude, longitude]], dtype=np.float32)

    global model
    preds = model.predict([arr, dummy_input])
    # If preds is a list, get the first element
    if isinstance(preds, list):
        preds = np.array(preds[0])
    else:
        preds = np.array(preds)
    score = float(preds.flatten()[0])
    is_counterfeit = bool(score > 0.5)
    confidence = score
    message = "Counterfeit" if is_counterfeit else "Authentic"

    return {
        "is_counterfeit": is_counterfeit,
        "confidence": confidence,
        "message": message
    }

@router.get("/predict/health")
async def predict_health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "model_loaded": True,
        "message": "Prediction endpoint is healthy",
    }



@router.post("/predict", response_model=ScanOut)
async def predict(
    file: UploadFile = File(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    brand: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    result = predict_image(file, latitude, longitude)
    scan = Scan(
        is_counterfeit=result["is_counterfeit"],
        confidence=result["confidence"],
        message=result["message"],
        latitude=latitude,
        longitude=longitude,
        brand=brand,
        date=datetime.utcnow()
    )
    db.add(scan)
    db.commit()
    db.refresh(scan)
    return scan

@router.get("/scans", response_model=list[ScanOut])
def get_scans(db: Session = Depends(get_db)):
    scans = db.query(Scan).all()
    return scans

@router.get("/api/locations", response_model=LocationsResponse)
async def get_locations(
    page: int = Query(1, ge=1),
    per_page: int = Query(100, le=500),
    db: Session = Depends(get_db),
):
    try:
        scans = (
            db.query(Scan)
            .filter(Scan.latitude != None, Scan.longitude != None)
            .offset((page - 1) * per_page)
            .limit(per_page)
            .all()
        )
        logger.info(f"Fetched {len(scans)} scans from database")
        locations = [
            {
                "id": scan.id,
                "lat": scan.latitude,
                "lng": scan.longitude,
                "is_counterfeit": scan.is_counterfeit,
                "brand": scan.brand,
                "confidence": f"{scan.confidence:.2%}",
                "date": f"{scan.date:%Y-%m-%d}",
            }
            for scan in scans
        ]
        return {"locations": locations}
    except Exception as e:
        logger.error(f"Location error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/Uploads/{filename}")
async def uploaded_file(filename: str):
    file_path = os.path.join(os.getenv("UPLOAD_FOLDER", "Uploads/"), filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)
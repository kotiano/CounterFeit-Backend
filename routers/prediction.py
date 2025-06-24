from fastapi import APIRouter, File, UploadFile, Depends, Query, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import text
from database import SessionLocal, get_db
from schemas import PredictRequest, PredictResponse, LocationsResponse
from repository.predict import process_prediction
from models import ScanResult
from logger_config import setup_logging
import os

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

@router.get("/predict/health")
async def predict_health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "model_loaded": True,
        "message": "Prediction endpoint is healthy",
    }

@router.post("/predict", response_model=PredictResponse)
async def predict(
    image: UploadFile = File(...),
    data: PredictRequest = Depends(),
    db: SessionLocal = Depends(get_db),
):
    return await process_prediction(image, data, db, model)

@router.get("/api/locations", response_model=LocationsResponse)
async def get_locations(
    page: int = Query(1, ge=1),
    per_page: int = Query(100, le=500),
    db: SessionLocal = Depends(get_db),
):
    try:
        scans = (
            db.query(ScanResult)
            .filter(ScanResult.latitude != None, ScanResult.longitude != None)
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
                "is_authentic": scan.is_authentic,
                "brand": scan.brand,
                "batch_no": scan.batch_no,
                "confidence": f"{scan.confidence:.2%}",
                "date": f"{scan.date:%Y-%m-%d}",
                "image_url": scan.image_url,
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
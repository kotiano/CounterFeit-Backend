import os
import uuid
from typing import Optional
import tensorflow as tf
import numpy as np
from fastapi import HTTPException
from dotenv import load_dotenv
from logger_config import setup_logging
from database import SessionLocal
from models import ScanResult
from schemas import LocationsResponse
from datetime import date, datetime
from sqlalchemy import text
from fastapi.responses import FileResponse
from repository.download_model import model


load_dotenv()

UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "Uploads/")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 5 * 1024 * 1024))
KEEP_IMAGES = os.getenv("KEEP_IMAGES", "false").lower() == "true"
LATENT_DIM = int(os.getenv("LATENT_DIM", 256))
SIGMA = float(os.getenv("SIGMA", 0.0003))
TARGET_SIZE = tuple(map(int, os.getenv("TARGET_SIZE", "224,224").split(",")))
AUTHENTICITY_THRESHOLD = float(os.getenv("AUTHENTICITY_THRESHOLD", 0.4))
MODEL_PATH = os.getenv("MODEL_PATH")

logger = setup_logging(__name__)


def initialize_model():
    try:
        if not os.path.exists(MODEL_PATH):
            logger.info(f"Model file not found at {MODEL_PATH}")
            gdrive_url = os.getenv("MODEL_GDRIVE_URL")
            if gdrive_url:
                logger.info("Attempting to download model from Google Drive")
                success = model(gdrive_url, MODEL_PATH)
                if not success:
                    logger.error("Failed to download model")
                    return None
            else:
                logger.error("MODEL_GDRIVE_URL not provided in .env")
                return None

        logger.info(f"Loading model from {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        logger.info(f"Model loaded successfully from {MODEL_PATH}")
        return model
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}", exc_info=True)
        return None

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def load_image(path: str) -> tf.Tensor:
    try:
        image = tf.io.read_file(path)
        if path.lower().endswith(".png"):
            image = tf.image.decode_png(image, channels=3)
        else:
            image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.adjust_contrast(image, contrast_factor=1.2)
        image = tf.image.resize(image, TARGET_SIZE)
        return image / 255.0
    except Exception as e:
        logger.error(f"Image load error: {e}")
        raise HTTPException(status_code=400, detail="Invalid image format")

def run_prediction(image_tensor: tf.Tensor, model) -> float:
    try:
        pseudo_negative = tf.random.normal([1, LATENT_DIM], mean=0.0, stddev=SIGMA)
        test_probs, _ = model.predict(
            [tf.expand_dims(image_tensor, axis=0), pseudo_negative], verbose=0, batch_size=1
        )
        return float(test_probs[0][0])
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

async def process_prediction(image, data: dict, db: SessionLocal, model) -> dict:
    filepath: Optional[str] = None
    try:
        if not allowed_file(image.filename):
            raise HTTPException(status_code=400, detail="Invalid file type")

        filename = f"{uuid.uuid4().hex}_{image.filename.lower()}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        content = await image.read()
        if len(content) > MAX_CONTENT_LENGTH:
            raise HTTPException(status_code=400, detail="File too large")
        with open(filepath, "wb") as f:
            f.write(content)

        image_tensor = load_image(filepath)
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        score = run_prediction(image_tensor, model)
        today = date.today()
        batch_no = f"{data['brand'][:3].upper()}-{datetime.now().year}-{uuid.uuid4().hex[:4]}"
        confidence = float(score)
        is_authentic = score >= AUTHENTICITY_THRESHOLD

        latitude = float(data['latitude']) if data.get('latitude') and data['latitude'] != "Unknown" else None
        longitude = float(data['longitude']) if data.get('longitude') and data['longitude'] != "Unknown" else None

        scan = ScanResult(
            brand=data['brand'],
            batch_no=batch_no,
            date=today,
            confidence=confidence,
            is_authentic=is_authentic,
            latitude=latitude,
            longitude=longitude,
            image_url=f"/Uploads/{filename}" if KEEP_IMAGES else None,
            timestamp=datetime.now().isoformat()
        )
        logger.info(f"Adding scan to database: {scan.__dict__}")
        db.add(scan)
        db.commit()
        logger.info(f"Scan committed to database: id={scan.id}")
        db.refresh(scan)
        logger.info(f"Scan refreshed from database: id={scan.id}")

        return {
            "id": scan.id,
            "is_authentic": is_authentic,
            "brand": data['brand'],
            "batch_no": batch_no,
            "date": f"{today:%Y-%m-%d}",
            "confidence": f"{confidence:.2%}",
            "latitude": str(latitude) if latitude is not None else "Unknown",
            "longitude": str(longitude) if longitude is not None else "Unknown",
            "image_url": scan.image_url or "",
            "message": "Authentic" if is_authentic else "Counterfeit detected",
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
    finally:
        if not KEEP_IMAGES and filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception as e:
                logger.error(f"Failed to delete file {filepath}: {e}")

def check_health(db: SessionLocal) -> dict:
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

def check_predict_health(model) -> dict:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "model_loaded": True,
        "message": "Prediction endpoint is healthy",
    }

def get_locations(page: int, per_page: int, db: SessionLocal) -> LocationsResponse:
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
                "image_url": scan.image_url or "",
                "timestamp": scan.timestamp.isoformat() if scan.timestamp else "",
            }
            for scan in scans
        ]
        return LocationsResponse(locations=locations)
    except Exception as e:
        logger.error(f"Location error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def get_uploaded_file(filename: str) -> FileResponse:
    file_path = os.path.join(os.getenv("UPLOAD_FOLDER", "Uploads/"), filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)
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
from datetime import date, datetime

load_dotenv()

UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "Uploads/")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 5 * 1024 * 1024))
KEEP_IMAGES = os.getenv("KEEP_IMAGES", "false").lower() == "true"
LATENT_DIM = int(os.getenv("LATENT_DIM", 256))
SIGMA = float(os.getenv("SIGMA", 0.0003))
TARGET_SIZE = tuple(map(int, os.getenv("TARGET_SIZE", "224,224").split(",")))
AUTHENTICITY_THRESHOLD = float(os.getenv("AUTHENTICITY_THRESHOLD", 0.4))

logger = setup_logging(__name__)

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
            [tf.expand_dims(image_tensor, axis=0), pseudo_negative], verbose=0
        )
        return float(test_probs[0][0])
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

async def process_prediction(
    image, data, db: SessionLocal, model
) -> dict:
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
        batch_no = f"{data.brand[:3].upper()}-{datetime.now().year}-{uuid.uuid4().hex[:4]}"
        confidence = float(score)
        is_authentic = score >= AUTHENTICITY_THRESHOLD

        scan = ScanResult(
            brand=data.brand,
            batch_no=batch_no,
            date=today,
            confidence=confidence,
            is_authentic=is_authentic,
            latitude=float(data.latitude) if data.latitude != "Unknown" else None,
            longitude=float(data.longitude) if data.longitude != "Unknown" else None,
            image_url=f"/Uploads/{filename}",
        )
        db.add(scan)
        db.commit()
        db.refresh(scan)

        return {
            "id": scan.id,
            "is_authentic": is_authentic,
            "brand": data.brand,
            "batch_no": batch_no,
            "date": f"{today:%Y-%m-%d}",
            "confidence": f"{confidence:.2%}",
            "latitude": data.latitude,
            "longitude": data.longitude,
            "image_url": f"/Uploads/{filename}",
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
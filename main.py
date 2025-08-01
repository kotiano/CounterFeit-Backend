import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from logger_config import setup_logging
from contextlib import asynccontextmanager
from database import Base, engine
from routers import prediction, admin
from repository.predict import UPLOAD_FOLDER
import tensorflow as tf

load_dotenv()
logger = setup_logging(__name__)

# Load model
MODEL_PATH = os.getenv("MODEL_PATH")
model = None
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    logger.info(f"Model loaded from {MODEL_PATH}")
    prediction.model = model  # Set global model for prediction router
except Exception as e:
    logger.error(f"Model loading failed: {e}")

# FastAPI app
app = FastAPI(title="Alcohol Detection API")

app.mount("/templates", StaticFiles(directory="templates"), name="templates")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://react-counterfeit.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(prediction.router)
app.include_router(admin.router)

# Create upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Database setup
Base.metadata.create_all(bind=engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starting up")
    logger.info(f"SECRET_KEY: {os.getenv('SECRET_KEY')}")
    required_env_vars = [
        "ADMIN_USERNAME",
        "ADMIN_PASSWORD_HASH",
        "MODEL_PATH",
        "UPLOAD_FOLDER",
        "ALLOWED_ORIGINS",
        "SECRET_KEY",
        "DATABASE_URI",
    ]
    for var in required_env_vars:
        if not os.getenv(var):
            raise ValueError(f"Missing required environment variable: {var}")
    # Check that the model file exists
    if not os.path.exists(os.getenv("MODEL_PATH")):
        raise ValueError(f"Model file not found: {os.getenv('MODEL_PATH')}")
    yield
    logger.info("Application shutting down")

app.lifespan = lifespan

@app.get("/")
async def home():
    return {"message": "Welcome to the Alcohol Detection API"}


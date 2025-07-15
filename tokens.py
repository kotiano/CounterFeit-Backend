from datetime import datetime, timedelta
from jose import JWTError, jwt
from dotenv import load_dotenv
from logger_config import setup_logging
import os

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
logger = setup_logging(__name__)

def create_access_token(data: dict) -> str:
    logger.info(f"Creating token with SECRET_KEY: {SECRET_KEY[:4]}...")
    to_encode = data.copy()
    expire = datetime.now() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str) -> str | None:
    logger.info(f"Verifying token with SECRET_KEY: {SECRET_KEY[:4]}...")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except JWTError as e:
        logger.error(f"JWT verification failed: {e}")
        return None
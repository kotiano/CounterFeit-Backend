from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from tokens import verify_token
from logger_config import setup_logging

logger = setup_logging(__name__)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/admin/login")

def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    username = verify_token(token)
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return username
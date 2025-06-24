from fastapi import APIRouter, Request, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from hashing import verify_password
from tokens import create_access_token
from oauth2 import get_current_user
from logger_config import setup_logging
import os

load_dotenv()

router = APIRouter(prefix="/admin", tags=["admin"])
templates = Jinja2Templates(directory="templates")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME")
ADMIN_PASSWORD_HASH = os.getenv("ADMIN_PASSWORD_HASH")
logger = setup_logging(__name__)

@router.get("/login", response_class=HTMLResponse)
async def admin_login_get(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@router.post("/login", response_class=HTMLResponse)
async def admin_login_post(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    if username == ADMIN_USERNAME and verify_password(password, ADMIN_PASSWORD_HASH):
        access_token = create_access_token({"sub": username})
        response = RedirectResponse(url="/admin/map", status_code=303)
        response.set_cookie(
            key="access_token",
            value=access_token,
            httponly=True,
            secure=False,  # For local development
            samesite="strict",
        )
        return response
    return templates.TemplateResponse(
        "login.html", {"request": request, "error": "Invalid credentials"}
    )

@router.get("/logout")
async def admin_logout():
    response = RedirectResponse(url="/admin/login", status_code=303)
    response.delete_cookie("access_token")
    return response

@router.get("/map", response_class=HTMLResponse)
async def admin_map(request: Request, current_user: str = Depends(get_current_user)):
    return templates.TemplateResponse("admin_dashboard.html", {"request": request})
from __future__ import annotations

import asyncio
import os
import re
import secrets
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Any, Optional

import bcrypt
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from jose import JWTError, jwt
from pydantic import BaseModel, Field, field_validator

import database
from gemini_utils import ai_enabled, get_home_recommendations, get_party_recommendations, get_jewelry_recommendations

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR/"static"/"uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

SECRET_KEY = os.getenv("SECRET_KEY") or secrets.token_urlsafe(48)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES","30"))
SESSION_IDLE_MINUTES = int(os.getenv("SESSION_IDLE_MINUTES","30"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB","5"))
ALLOWED_ORIGINS = [x.strip() for x in os.getenv(
    "ALLOWED_ORIGINS","http://127.0.0.1:8000,http://localhost:8000"
).split(",") if x.strip()]

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token", auto_error=False)
templates = Jinja2Templates(directory=str(BASE_DIR/"templates"))

class RegisterUser(BaseModel):
    username: str = Field(min_length=3, max_length=30)
    email: str = Field(min_length=5, max_length=120)
    full_name: str = Field(min_length=2, max_length=80)
    password: str = Field(min_length=8, max_length=128)
    @field_validator("username")
    @classmethod
    def username_ok(cls, v: str) -> str:
        v = v.strip().lower()
        if not re.fullmatch(r"[a-z0-9_.-]+", v):
            raise ValueError("Username may contain letters, numbers, dot, dash and underscore only")
        return v
    @field_validator("email")
    @classmethod
    def email_ok(cls, v: str) -> str:
        v = v.strip().lower()
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", v):
            raise ValueError("Enter a valid email address")
        return v

class Token(BaseModel):
    access_token: str
    token_type: str

class HomeBudgetInput(BaseModel):
    total_budget: float = Field(gt=0, le=10_000_000)
    num_lights: int = Field(ge=0, le=100)
    num_fans: int = Field(ge=0, le=100)
    num_furniture: int = Field(ge=0, le=100)
    num_dining_tables: int = Field(ge=0, le=20)
    has_living_room: bool = False
    has_kitchen: bool = False
    has_bedroom: bool = False
    additional_requirements: str = Field(default="", max_length=1000)

class PartyBudgetInput(BaseModel):
    total_budget: float = Field(gt=0, le=100_000_000)
    party_type: str = Field(min_length=2, max_length=80)
    num_guests: int = Field(gt=0, le=100_000)
    venue_type: str = Field(default="", max_length=120)
    needs_catering: bool = True
    needs_decoration: bool = True
    needs_entertainment: bool = True
    additional_requirements: str = Field(default="", max_length=1000)

class JewelryBudgetInput(BaseModel):
    total_budget: float = Field(gt=0, le=100_000_000)
    occasion: str = Field(min_length=2, max_length=80)
    preferences: str = Field(default="", max_length=1000)

class SessionDataInput(BaseModel):
    user_data: dict[str, Any]

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except ValueError:
        return False

def authenticate_user(username: str, password: str) -> dict[str, Any] | None:
    user = database.get_user(username.strip().lower())
    return user if user and verify_password(password, user["hashed_password"]) else None

def create_access_token(username: str) -> tuple[str,str,datetime]:
    expires = utc_now()+timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    token_id = str(uuid.uuid4())
    token = jwt.encode({"sub":username,"jti":token_id,"exp":expires}, SECRET_KEY, algorithm=ALGORITHM)
    return token, token_id, expires

def _extract_token(request: Request, bearer: Optional[str]) -> str | None:
    return bearer or request.cookies.get("access_token")

def _decode_and_validate(token: str) -> tuple[dict[str, Any],dict[str, Any]]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username, token_id = payload.get("sub"), payload.get("jti")
        if not username or not token_id:
            raise HTTPException(401, "Invalid authentication token")
    except JWTError as exc:
        raise HTTPException(401, "Invalid or expired authentication token") from exc
    session = database.get_session(token_id)
    if not session or session["revoked"]:
        raise HTTPException(401, "Session is not active")
    last = datetime.fromisoformat(session["last_activity"])
    if utc_now()-last > timedelta(minutes=SESSION_IDLE_MINUTES):
        database.revoke_session(token_id)
        raise HTTPException(401, "Session expired due to inactivity")
    user = database.get_user(username)
    if not user:
        raise HTTPException(401, "User not found")
    database.touch_session(token_id)
    session["token_id"] = token_id
    return user, session

async def get_current_user(request: Request, bearer: Annotated[Optional[str],Depends(oauth2_scheme)]=None):
    token = _extract_token(request, bearer)
    if not token:
        raise HTTPException(401, "Not authenticated")
    return _decode_and_validate(token)[0]

async def get_current_session(request: Request, bearer: Annotated[Optional[str],Depends(oauth2_scheme)]=None):
    token = _extract_token(request, bearer)
    if not token:
        raise HTTPException(401, "Not authenticated")
    return _decode_and_validate(token)[1]

def optional_user(request: Request):
    token = request.cookies.get("access_token")
    if not token:
        return None
    try:
        return _decode_and_validate(token)[0]
    except HTTPException:
        return None

def page_guard(request: Request):
    user = optional_user(request)
    return user if user else RedirectResponse("/login", status_code=303)

def save_history(username: str, kind: str, input_data: dict[str,Any], result: dict[str,Any]) -> str:
    rec_id = str(uuid.uuid4())
    database.save_recommendation(rec_id, username, kind, input_data, {
        "total_budget":result.get("total_budget",0),
        "remaining_budget":result.get("remaining_budget",0),
        "ai_mode":result.get("ai_mode","unknown"),
    }, result)
    return rec_id

async def cleanup_loop():
    while True:
        database.cleanup_sessions((utc_now()-timedelta(minutes=SESSION_IDLE_MINUTES)).isoformat())
        await asyncio.sleep(300)

@asynccontextmanager
async def lifespan(app: FastAPI):
    database.init_db()
    task = asyncio.create_task(cleanup_loop())
    yield
    task.cancel()

app = FastAPI(title="PocketSmart AI", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(BASE_DIR/"static")), name="static")

@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException):
    if exc.status_code == 401 and "text/html" in request.headers.get("accept",""):
        return RedirectResponse("/login", status_code=303)
    return JSONResponse(status_code=exc.status_code, content={"detail":exc.detail})

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html",
        context={"user":optional_user(request),"ai_enabled":ai_enabled()})

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    if optional_user(request):
        return RedirectResponse("/dashboard",303)
    return templates.TemplateResponse(request=request,name="register.html",context={"error":None})

@app.post("/register")
async def register(request: Request, username: Annotated[str,Form()], email: Annotated[str,Form()],
                   full_name: Annotated[str,Form()], password: Annotated[str,Form()]):
    try:
        data = RegisterUser(username=username,email=email,full_name=full_name,password=password)
    except Exception as exc:
        return templates.TemplateResponse(request=request,name="register.html",
            context={"error":str(exc)},status_code=422)
    if database.get_user(data.username):
        return templates.TemplateResponse(request=request,name="register.html",
            context={"error":"Username already exists."},status_code=409)
    if database.get_user_by_email(data.email):
        return templates.TemplateResponse(request=request,name="register.html",
            context={"error":"Email is already registered."},status_code=409)
    database.create_user(data.username,data.email,data.full_name,hash_password(data.password))
    return RedirectResponse("/login?registered=1",303)

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if optional_user(request):
        return RedirectResponse("/dashboard",303)
    return templates.TemplateResponse(request=request,name="login.html",
        context={"error":None,"registered":request.query_params.get("registered")=="1"})

@app.post("/login")
async def login(request: Request, username: Annotated[str,Form()], password: Annotated[str,Form()]):
    user = authenticate_user(username,password)
    if not user:
        return templates.TemplateResponse(request=request,name="login.html",
            context={"error":"Incorrect username or password.","registered":False},status_code=401)
    token, token_id, expires = create_access_token(user["username"])
    now = utc_now().isoformat()
    database.create_session(token_id,user["username"],now,now,expires.isoformat())
    response = RedirectResponse("/dashboard",303)
    response.set_cookie("access_token",token,httponly=True,samesite="lax",secure=False,
                        max_age=ACCESS_TOKEN_EXPIRE_MINUTES*60)
    return response

@app.post("/token", response_model=Token)
async def token(form_data: Annotated[OAuth2PasswordRequestForm,Depends()]):
    user = authenticate_user(form_data.username,form_data.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Incorrect username or password",
                            headers={"WWW-Authenticate":"Bearer"})
    value, token_id, expires = create_access_token(user["username"])
    now = utc_now().isoformat()
    database.create_session(token_id,user["username"],now,now,expires.isoformat())
    return {"access_token":value,"token_type":"bearer"}

@app.post("/logout")
async def logout(request: Request):
    value = request.cookies.get("access_token")
    if value:
        try:
            payload = jwt.decode(value,SECRET_KEY,algorithms=[ALGORITHM])
            if payload.get("jti"):
                database.revoke_session(payload["jti"])
        except JWTError:
            pass
    response = RedirectResponse("/login",303)
    response.delete_cookie("access_token")
    return response

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    user = page_guard(request)
    if isinstance(user,RedirectResponse):
        return user
    return templates.TemplateResponse(request=request,name="dashboard.html",
        context={"user":user,"history":database.list_recommendations(user["username"])[:3],
                 "ai_enabled":ai_enabled()})

@app.get("/home-planner", response_class=HTMLResponse)
async def home_planner(request: Request):
    user = page_guard(request)
    if isinstance(user,RedirectResponse): return user
    return templates.TemplateResponse(request=request,name="home_planner.html",context={"user":user})

@app.get("/party-planner", response_class=HTMLResponse)
async def party_planner(request: Request):
    user = page_guard(request)
    if isinstance(user,RedirectResponse): return user
    return templates.TemplateResponse(request=request,name="party_planner.html",context={"user":user})

@app.get("/jewelry-planner", response_class=HTMLResponse)
async def jewelry_planner(request: Request):
    user = page_guard(request)
    if isinstance(user,RedirectResponse): return user
    return templates.TemplateResponse(request=request,name="jewelry_planner.html",context={"user":user})

@app.post("/generate-home")
async def generate_home_json(payload: HomeBudgetInput,
    current_user: Annotated[dict[str,Any],Depends(get_current_user)]):
    result = get_home_recommendations(payload.model_dump())
    rec_id = save_history(current_user["username"],"home",payload.model_dump(),result)
    return {"recommendation_id":rec_id,**result}

@app.post("/home-budget", response_class=HTMLResponse)
async def home_budget(request: Request, current_user: Annotated[dict[str,Any],Depends(get_current_user)],
    total_budget: Annotated[float,Form()], num_lights: Annotated[int,Form()]=0,
    num_fans: Annotated[int,Form()]=0, num_furniture: Annotated[int,Form()]=0,
    num_dining_tables: Annotated[int,Form()]=0, has_living_room: Annotated[bool,Form()]=False,
    has_kitchen: Annotated[bool,Form()]=False, has_bedroom: Annotated[bool,Form()]=False,
    additional_requirements: Annotated[str,Form()]=""):
    payload = HomeBudgetInput(total_budget=total_budget,num_lights=num_lights,num_fans=num_fans,
        num_furniture=num_furniture,num_dining_tables=num_dining_tables,
        has_living_room=has_living_room,has_kitchen=has_kitchen,has_bedroom=has_bedroom,
        additional_requirements=additional_requirements)
    result = get_home_recommendations(payload.model_dump())
    rec_id = save_history(current_user["username"],"home",payload.model_dump(),result)
    return templates.TemplateResponse(request=request,name="result.html",
        context={"user":current_user,"result":result,"kind":"Home Interior","recommendation_id":rec_id})

@app.post("/generate-party")
async def generate_party_json(payload: PartyBudgetInput,
    current_user: Annotated[dict[str,Any],Depends(get_current_user)]):
    result = get_party_recommendations(payload.model_dump())
    rec_id = save_history(current_user["username"],"party",payload.model_dump(),result)
    return {"recommendation_id":rec_id,**result}

@app.post("/party-budget", response_class=HTMLResponse)
async def party_budget(request: Request, current_user: Annotated[dict[str,Any],Depends(get_current_user)],
    total_budget: Annotated[float,Form()], party_type: Annotated[str,Form()],
    num_guests: Annotated[int,Form()], venue_type: Annotated[str,Form()]="",
    needs_catering: Annotated[bool,Form()]=False, needs_decoration: Annotated[bool,Form()]=False,
    needs_entertainment: Annotated[bool,Form()]=False,
    additional_requirements: Annotated[str,Form()]=""):
    payload = PartyBudgetInput(total_budget=total_budget,party_type=party_type,num_guests=num_guests,
        venue_type=venue_type,needs_catering=needs_catering,needs_decoration=needs_decoration,
        needs_entertainment=needs_entertainment,additional_requirements=additional_requirements)
    result = get_party_recommendations(payload.model_dump())
    rec_id = save_history(current_user["username"],"party",payload.model_dump(),result)
    return templates.TemplateResponse(request=request,name="result.html",
        context={"user":current_user,"result":result,"kind":"Party","recommendation_id":rec_id})

def validate_image(upload: UploadFile):
    if upload.content_type not in {"image/jpeg","image/png","image/webp"}:
        raise HTTPException(400,"Upload a JPEG, PNG, or WebP image.")

async def save_upload(upload: UploadFile) -> Path:
    validate_image(upload)
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in {".jpg",".jpeg",".png",".webp"}: suffix=".jpg"
    path = UPLOAD_DIR/f"{uuid.uuid4()}{suffix}"
    total = 0
    with path.open("wb") as out:
        while chunk := await upload.read(1024*1024):
            total += len(chunk)
            if total > MAX_UPLOAD_MB*1024*1024:
                path.unlink(missing_ok=True)
                raise HTTPException(413,f"Image must be <= {MAX_UPLOAD_MB} MB.")
            out.write(chunk)
    return path

@app.post("/generate-jewelry")
async def generate_jewelry_json(current_user: Annotated[dict[str,Any],Depends(get_current_user)],
    total_budget: Annotated[float,Form()], occasion: Annotated[str,Form()],
    preferences: Annotated[str,Form()]="",
    outfit_image: Annotated[Optional[UploadFile],File()]=None):
    payload = JewelryBudgetInput(total_budget=total_budget,occasion=occasion,preferences=preferences)
    image_path = None
    try:
        if outfit_image and outfit_image.filename: image_path = await save_upload(outfit_image)
        result = get_jewelry_recommendations(payload.model_dump(),str(image_path) if image_path else None)
        rec_id = save_history(current_user["username"],"jewelry",payload.model_dump(),result)
        return {"recommendation_id":rec_id,**result}
    finally:
        if image_path: image_path.unlink(missing_ok=True)

@app.post("/jewelry-budget", response_class=HTMLResponse)
async def jewelry_budget(request: Request,current_user: Annotated[dict[str,Any],Depends(get_current_user)],
    total_budget: Annotated[float,Form()], occasion: Annotated[str,Form()],
    preferences: Annotated[str,Form()]="",
    outfit_image: Annotated[Optional[UploadFile],File()]=None):
    payload = JewelryBudgetInput(total_budget=total_budget,occasion=occasion,preferences=preferences)
    image_path=None
    try:
        if outfit_image and outfit_image.filename: image_path=await save_upload(outfit_image)
        result=get_jewelry_recommendations(payload.model_dump(),str(image_path) if image_path else None)
        rec_id=save_history(current_user["username"],"jewelry",payload.model_dump(),result)
        return templates.TemplateResponse(request=request,name="result.html",
            context={"user":current_user,"result":result,"kind":"Jewelry","recommendation_id":rec_id})
    finally:
        if image_path: image_path.unlink(missing_ok=True)

@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request):
    user=page_guard(request)
    if isinstance(user,RedirectResponse): return user
    return templates.TemplateResponse(request=request,name="history.html",
        context={"user":user,"records":database.list_recommendations(user["username"])})

@app.get("/recommendation-history")
async def recommendation_history(current_user: Annotated[dict[str,Any],Depends(get_current_user)]):
    return database.list_recommendations(current_user["username"])

@app.get("/recommendation-details/{recommendation_id}")
async def recommendation_details(recommendation_id: str,
    current_user: Annotated[dict[str,Any],Depends(get_current_user)]):
    record=database.get_recommendation(recommendation_id,current_user["username"])
    if not record: raise HTTPException(404,"Recommendation not found")
    return record

@app.get("/history/{recommendation_id}", response_class=HTMLResponse)
async def history_detail(request: Request,recommendation_id: str):
    user=page_guard(request)
    if isinstance(user,RedirectResponse): return user
    record=database.get_recommendation(recommendation_id,user["username"])
    if not record: raise HTTPException(404,"Recommendation not found")
    return templates.TemplateResponse(request=request,name="history_detail.html",
        context={"user":user,"record":record,"result":record["full_result"]})

@app.get("/session-info")
async def session_info(current_user: Annotated[dict[str,Any],Depends(get_current_user)],
    session: Annotated[dict[str,Any],Depends(get_current_session)]):
    return {"username":current_user["username"],"login_time":session["login_time"],
            "last_activity":session["last_activity"],"expires_at":session["expires_at"],
            "user_data":session["user_data"]}

@app.post("/session-data")
async def session_data(payload: SessionDataInput,
    session: Annotated[dict[str,Any],Depends(get_current_session)]):
    database.update_session_data(session["token_id"],payload.user_data)
    return {"message":"Session data updated","user_data":payload.user_data}

@app.get("/health")
async def health():
    return {"status":"ok","ai_enabled":ai_enabled()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app",host="0.0.0.0",port=8000,reload=True)

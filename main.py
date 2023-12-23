from fastapi import FastAPI, HTTPException, Query, UploadFile, File as FastAPIFile, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from jose import JWTError, jwt
from datetime import datetime, timedelta
from passlib.context import CryptContext
from openai import OpenAI, OpenAIError
from fastapi.middleware.cors import CORSMiddleware
import os
from sqlalchemy.orm import Session
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt
from sqlalchemy.orm import Session

# 環境変数からシークレットキーと有効期限を取得
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "default_secret_key")  # デフォルト値を提供する
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", 30))  # 環境変数が設定されていない場合のデフォルト値


# データベース接続とモデル定義
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

# データベースモデル
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)

class Assistant(Base):
    __tablename__ = "assistants"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    model = Column(String)

class Thread(Base):
    __tablename__ = "threads"
    id = Column(Integer, primary_key=True, index=True)
    assistant_id = Column(Integer, ForeignKey('assistants.id'))
    assistant = relationship("Assistant")

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    thread_id = Column(Integer, ForeignKey('threads.id'))
    thread = relationship("Thread")
    content = Column(String)
    role = Column(String)

class FileModel(Base):
    __tablename__ = "files"
    id = Column(Integer, primary_key=True, index=True)
    file_content = Column(LargeBinary)
    purpose = Column(String)

# FastAPIアプリケーションの初期化
app = FastAPI()

# CORSミドルウェアの設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # すべてのオリジンを許可
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# APIクライアントの設定
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# 依存関係
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user

# パスワードハッシュ化の設定
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# パスワードをハッシュ化する関数
def hash_password(password: str):
    return pwd_context.hash(password)

# パスワードの検証
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

# JWTトークン生成用の関数
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# モデル定義
class CreateAssistantRequest(BaseModel):
    name: str
    instructions: str
    model: str = "gpt-3.5-turbo-1106"
    tools: list = [{"type": "code_interpreter"}]

class CreateThreadRequest(BaseModel):
    pass

class AddMessageRequest(BaseModel):
    thread_id: str
    role: str
    content: str

class RunAssistantRequest(BaseModel):
    thread_id: str
    assistant_id: str

# ユーザー作成のエンドポイント
@app.post("/users/")
def create_user(username: str, email: str, password: str, db: Session = Depends(get_db)):
    hashed_password = hash_password(password)
    db_user = User(username=username, email=email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.post("/token")
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me")
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

@app.post("/create_assistant")
def create_assistant(request: CreateAssistantRequest, db: Session = Depends(get_db)):
    assistant = Assistant(name=request.name, model=request.model)
    db.add(assistant)
    db.commit()
    db.refresh(assistant)
    return {"assistant_id": assistant.id}

@app.post("/create_thread")
def create_thread(db: Session = Depends(get_db)):
    thread = Thread()
    db.add(thread)
    db.commit()
    db.refresh(thread)
    return {"thread_id": thread.id}

@app.post("/add_message")
def add_message(request: AddMessageRequest, db: Session = Depends(get_db)):
    message = Message(thread_id=request.thread_id, content=request.content, role=request.role)
    db.add(message)
    db.commit()
    db.refresh(message)
    return {"message_id": message.id}

@app.post("/run_assistant")
def run_assistant(request: RunAssistantRequest):
    try:
        run = client.beta.threads.runs.create(
            thread_id=request.thread_id,
            assistant_id=request.assistant_id
        )
        return {"run_id": run.id}
    except OpenAIError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {e}")

@app.get("/check_run_status")
def check_run_status(thread_id: str = Query(...), run_id: str = Query(...)):
    try:
        run_status = client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run_id
        )
        return {"status": run_status.status}
    except OpenAIError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {e}")

@app.get("/get_responses")
def get_responses(thread_id: str = Query(...), db: Session = Depends(get_db)):
    responses = db.query(Message).filter(Message.thread_id == thread_id, Message.role == "assistant").all()
    return {"responses": [response.content for response in responses]}

@app.post("/upload_file")
async def upload_file(file: UploadFile = FileModel(...), db: Session = Depends(get_db)):
    file_content = await file.read()
    db_file = FileModel(file_content=file_content, purpose='assistants')
    db.add(db_file)
    db.commit()
    db.refresh(db_file)
    return {"file_id": db_file.id}
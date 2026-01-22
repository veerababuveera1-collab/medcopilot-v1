import os, shutil, uuid, json
from datetime import datetime, timedelta
from typing import List

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from jose import jwt, JWTError
from passlib.context import CryptContext

from sqlalchemy import create_engine, Column, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from pypdf import PdfReader
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# =====================================================
# CONFIG
# =====================================================

SECRET_KEY = "MEDINTEL_SECRET_KEY"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

UPLOAD_DIR = "uploads"
DB_DIR = "database"
VECTOR_INDEX = f"{DB_DIR}/index.faiss"
META_FILE = f"{DB_DIR}/meta.json"
AUDIT_DB = "sqlite:///./audit.db"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# =====================================================
# APP
# =====================================================

app = FastAPI(title="MEDINTEL AI Enterprise Backend", version="1.0")

# =====================================================
# SECURITY
# =====================================================

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def hash_password(password: str):
    return pwd_context.hash(password)

def verify_password(password: str, hashed: str):
    return pwd_context.verify(password, hashed)

def create_access_token(data: dict):
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    data.update({"exp": expire})
    return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

# =====================================================
# USERS
# =====================================================

USERS = {
    "admin": {"password": hash_password("admin123"), "role": "ADMIN"},
    "doctor": {"password": hash_password("doctor123"), "role": "REVIEWER"}
}

# =====================================================
# DATABASE (AUDIT)
# =====================================================

engine = create_engine(AUDIT_DB, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(String, primary_key=True)
    user = Column(String)
    action = Column(String)
    details = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

def audit(db: Session, user: str, action: str, details: str):
    db.add(AuditLog(
        id=str(uuid.uuid4()),
        user=user,
        action=action,
        details=details
    ))
    db.commit()

# =====================================================
# AUTH
# =====================================================

class Token(BaseModel):
    access_token: str
    token_type: str

@app.post("/token", response_model=Token)
def login(username: str, password: str):
    user = USERS.get(username)
    if not user or not verify_password(password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"sub": username, "role": user["role"]})
    return {"access_token": token, "token_type": "bearer"}

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username not in USERS:
            raise HTTPException(status_code=401)
        return {"username": username, "role": USERS[username]["role"]}
    except JWTError:
        raise HTTPException(status_code=401)

# =====================================================
# AI VECTOR ENGINE (LOCAL)
# =====================================================

model = SentenceTransformer("all-MiniLM-L6-v2")
DIM = 384

if os.path.exists(VECTOR_INDEX) and os.path.exists(META_FILE):
    index = faiss.read_index(VECTOR_INDEX)
    metadata = json.load(open(META_FILE))
else:
    index = faiss.IndexFlatL2(DIM)
    metadata = []

def save_index():
    faiss.write_index(index, VECTOR_INDEX)
    json.dump(metadata, open(META_FILE, "w"), indent=2)

def read_pdf(path):
    reader = PdfReader(path)
    text = ""
    for p in reader.pages:
        if p.extract_text():
            text += p.extract_text()
    return text

def chunk_text(text, size=500):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

def index_document(text, source):
    chunks = chunk_text(text)
    embeddings = model.encode(chunks)
    index.add(np.array(embeddings).astype("float32"))
    for c in chunks:
        metadata.append({"text": c, "source": source})
    save_index()

def search(query, k=5):
    q = model.encode([query]).astype("float32")
    _, ids = index.search(q, k)
    return [metadata[i] for i in ids[0] if i < len(metadata)]

# =====================================================
# API MODELS
# =====================================================

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

# =====================================================
# ENDPOINTS
# =====================================================

@app.post("/upload")
def upload_docs(
    files: List[UploadFile] = File(...),
    user=Depends(get_current_user),
    db: Session = Depends(SessionLocal)
):
    if user["role"] != "ADMIN":
        raise HTTPException(status_code=403, detail="Only ADMIN can upload")

    for f in files:
        path = os.path.join(UPLOAD_DIR, f.filename)
        with open(path, "wb") as buffer:
            shutil.copyfileobj(f.file, buffer)
        text = read_pdf(path)
        index_document(text, f.filename)

    audit(db, user["username"], "UPLOAD", "Documents indexed")
    return {"status": "Indexed successfully"}

@app.post("/ask", response_model=QueryResponse)
def ask(
    req: QueryRequest,
    user=Depends(get_current_user),
    db: Session = Depends(SessionLocal)
):
    results = search(req.question)
    if not results:
        raise HTTPException(status_code=404, detail="No documents indexed")

    answer = "\n".join([r["text"][:300] for r in results])
    sources = list({r["source"] for r in results})

    audit(db, user["username"], "QUERY", req.question)
    return QueryResponse(answer=answer, sources=sources)

@app.get("/health")
def health():
    return {"status": "MEDINTEL AI Engine Running"}

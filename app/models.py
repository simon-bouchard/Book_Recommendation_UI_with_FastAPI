from pydantic import BaseModel, Field, EmailStr
from datetime import datetime
from typing import Optional
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=['bcrypt'], deprecated='auto')

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

class UserSignup(BaseModel):
    location: str 
    username: str 
    email: EmailStr = Field(default='unknown@example.com')
    created_at: datetime = Field(default_factory=datetime.utcnow)
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class Rating(BaseModel):
    user_id: str
    isbn: str
    rating: int = Field(ge=0, le=10)
    timestamp: datetime = Field(dafault_factory=datetime.utcnow)

class Book(BaseModel):
    id: str = Field(..., alias='_id')


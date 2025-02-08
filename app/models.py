from pydantic import BaseModel, Field, EmailStr
from datetime import datetime
from typing import Optional

class User(BaseModel):
    location: str 
    name: str = Field(default='Anonymous')
    email: EmailStr = Field(default='unknown@example.com')
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Rating(BaseModel):
    user_id: str
    isbn: str
    rating: int = Field(ge=0, le=10)
    timestamp: datetime = Field(dafault_factory=datetime.utcnow)

class Book(BaseModel):
    id: str = Field(..., alias='_id')


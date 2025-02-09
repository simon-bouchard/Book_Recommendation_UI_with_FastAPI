from fastapi import FastAPI, HTTPException, Depends
from models import UserSignup, hash_password
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

load_dotenv()

client = AsyncIOMotorClient(os.getenv('MONGO_URI'))
db = client['book-recommendation']
users = db['users']

app = FastAPI()

@routes.post('/signup')
async def signup(user: UserSignup):
    existing_user = await users.findOne({'username': user.username})

    if existing_user:
        raise HHTPException(status_code=400, detail='Username already taken')

    hashed_password = hash_password(user.password)
    new_user = {'username': user.username, 'email', user.email, 'password': hashed_password, 'location': user.location}

    await users.insert_one(new_user)

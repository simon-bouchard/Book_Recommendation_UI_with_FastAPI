import pandas as pd
from fastapi import APIRouter, HTTPException, Form, Request, FastAPI, Depends, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import jwt
from datetime import datetime, timedelta
from app.auth import get_current_user

router = APIRouter()
templates = Jinja2Templates(directory="templates")

BOOKS_CSV = 'data/BX-Books.csv'

try:
    df = pd.read_csv(BOOKS_CSV, encoding='ISO-8859-1', sep=';', usecols=['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher'])
    df.columns = df.columns.str.strip()
    books_data = df.set_index('ISBN').to_dict(orient='index')
except Exception as e:
    print(f'Error loading books: {e}')
    books_data = {}

load_dotenv()
client = MongoClient(os.getenv('MONGO_URI'))
db = client['book-recommendation']
ratings = db['ratings']

@router.get('/login', response_class=HTMLResponse)
def signup_page(request: Request):
    return templates.TemplateResponse('login.html', {'request': request})

@router.get('/profile')
def profile_page(request: Request, current_user: dict = Depends(get_current_user)):
    if not current_user:
        return RedirectResponse(ufl='/login', status_code=status.HTTP_303_SEE_OTHER)
    return templates.TemplateResponse('profile.html', {'request': request, 'user': current_user})

@router.post('/rating')
async def new_rating(current_user: dict = Depends(get_current_user), isbn: str = Form(...), comment: str = Form(...), rating: int = Form(...)):

    if not current_user:
        return RedirectResponse(url='/login', status_code=status.HTTP_303_SEE_OTHER)

    await ratings.update_one(
        {'user_id': current_user['_id'], 'isbn': isbn}, 
        {"$set": {'rating': rating, 'comment': comment}},
        upsert=True
    )

    return {'message': f'Rating successfully submitted/updated!'}

@router.get('/book/{isbn}', response_class=HTMLResponse)
async def book_recommendation(request: Request, isbn: str): 
    book = books_data.get(isbn)

    pipeline = [
        {"$match": {"isbn": isbn}},
        {"$group": {"_id": "$isbn", "avg_rating": {"$avg": "$rating"}}}
    ]

    average = round(list(ratings.aggregate(pipeline))[1]['avg_rating'], 2)

    if not book: 
        raise HTTPException(status_code=404, detail='Book not found')

    book = {
            'isbn': isbn,
            'title': book['Book-Title'],
            'author': book['Book-Author'],
            'year': book['Year-Of-Publication'],
            'publisher': book['Publisher'],
            'average_rating': average
    }
    
    return templates.TemplateResponse('book.html', {"request": request, "book": book})

@router.get('/logout')
async def logout():
    response = RedirectResponse(url='/login', status_code=HTTP_303_SEE_OTHER)

    response.delete_cookie(key='access_token')

    return response


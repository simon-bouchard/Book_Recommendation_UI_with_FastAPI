import pandas as pd
from fastapi import APIRouter, HTTPException, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pymongo import MongoClient
import os
from dotenv import load_dotenv

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

@router.get('/book/{isbn}', response_class=HTMLResponse)
async def book_recommendation(request: Request, isbn: str): 
    book = books_data.get(isbn)

    pipeline = [
        {"$match": {"isbn": isbn}},
        {"$group": {"_id": "$isbn", "avg_rating": {"$avg": "$rating"}}}
    ]

    average = round(list(ratings.aggregate(pipeline))[0]['avg_rating'], 2)

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

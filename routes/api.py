import pandas as pd
from fastapi import APIRouter, HTTPException, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

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

print(list(books_data.items())[:5])

@router.post('/book/{isbn}', response_class=HTMLResponse)
def book_recommendation(request: Request, isbn: str): 
    book = books_data.get(isbn)

    if not book: 
        raise HTTPException(status_code=404, detail='Book not found')
    book = {
            'isbn': isbn,
            'title': book['Book-Title'],
            'author': book['Book-Author'],
            'year': book['Year-Of-Publication'],
            'publisher': book['Publisher']
    }
    
    return templates.TemplateResponse('book.html', {"request": request, "book": book})

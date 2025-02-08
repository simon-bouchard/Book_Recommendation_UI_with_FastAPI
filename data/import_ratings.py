import pandas as pd
from pymongo import MongoClient
import os
import sys
from dotenv import load_dotenv
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.models import Rating

load_dotenv()

client = MongoClient(os.getenv('MONGO_URI'))

db = client['book-recommendation']

ratings = db['ratings']

file_path = os.path.join(os.getcwd(), 'BX-Book-Ratings.csv')

df = pd.read_csv('BX-Book-Ratings.csv', encoding='ISO-8859-1', sep=';')

df.rename(columns={
    'User-ID': 'user_id',
    'ISBN': 'isbn',
    'Book-Rating': 'rating'
}, inplace=True)

df['timestamp'] = datetime.utcnow()

df['rating'] = df['rating'].astype(int)

df['user_id'] = df['user_id'].astype(str)

data = df.to_dict(orient='records')

validated_ratings = [Rating(**rating).dict(by_alias=True) for rating in data]

ratings.insert_many(validated_ratings)

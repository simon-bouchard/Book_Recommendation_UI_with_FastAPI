import pandas as pd
from pymongo import MongoClient
import os
import sys
from dotenv import load_dotenv
from datetime import datetime
from bson import ObjectId

load_dotenv()

client = MongoClient(os.getenv('MONGO_URI'))

db = client['book-recommendation']

ratings = db['ratings']
users = db['users']

file_path = os.path.join(os.getcwd(), 'BX-Book-Ratings.csv')

df = pd.read_csv('BX-Book-Ratings.csv', encoding='ISO-8859-1', sep=';')

df.rename(columns={
    'User-ID': 'old_id',
    'ISBN': 'isbn',
    'Book-Rating': 'rating'
}, inplace=True)

user_mapping = {user['old_id']: user['_id'] for user in users.find({}, {'_id': 1, 'old_id': 1}) if 'old_id' in user}

df['user_id'] = df['old_id'].map(user_mapping)
df.dropna(subset=['user_id'], inplace=True)
df.drop(columns=['old_id'], inplace=True)

df['created_at'] = datetime.utcnow()

df['rating'] = df['rating'].astype(int)

df['user_id'] = df['user_id'].apply(lambda x: ObjectId(x) if pd.notna(x) else None)

data = df.to_dict(orient='records')

ratings.insert_many(data)

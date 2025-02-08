import pandas as pd
from pymongo import MongoClient
import os
import sys
from dotenv import load_dotenv
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.models import User

load_dotenv()

client = MongoClient(os.getenv('MONGO_URI'))

db = client['book-recommendation']

users = db['users']

df = pd.read_csv('BX-Users.csv', encoding='ISO-8859-1', sep=';')

df.rename(columns={
    'User-ID': '_id',
    'Location': 'location',
    'Age': 'age'
}, inplace=True)

df['created_at'] = datetime.utcnow()

df['_id'] = df['_id'].astype(str)

data = df.to_dict(orient='records')

users.insert_many(data, ordered=True)

import pandas as pd
from pymongo import MongoClient
import os
import sys
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

client = MongoClient(os.getenv('MONGO_URI'))

db = client['book-recommendation']

users = db['users']

df = pd.read_csv('./BX-Users.csv', encoding='ISO-8859-1', sep=';')

df.rename(columns={
    'User-ID': 'old_id',
}, inplace=True)

df = df.drop(columns=['Age', 'Location'])

df['created_at'] = datetime.utcnow()

df['email'] = 'unknown@example.com'

data = df.to_dict(orient='records')

users.insert_many(data, ordered=True)

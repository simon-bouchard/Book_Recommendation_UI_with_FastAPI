import requests
import random
import time
import os
from dotenv import load_dotenv
import sys
from motor.motor_asyncio import AsyncIOMotorClient
import asyncio
from bson import ObjectId
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.user_model import get_user_recommendations
from app.auth import get_current_user

load_dotenv()
BASE_URL = os.getenv('BASE_URL')

client = AsyncIOMotorClient(os.getenv('MONGO_URI'))
db = client['book-recommendation']
books = db['Books']
ratings = db['ratings']

def create_user(username):
    url = f'{BASE_URL}/auth/signup'
    payload = {'username': username, 'email': f'{username}@example.com', 'password': 'test'}

    session = requests.Session()
    response = session.post(url, data=payload, allow_redirects=False)

    print(f"Signup Response: {response.status_code} - {response.text}")

    if response.status_code == 200:
        user_data = response.json()
        user_id = user_data.get('user_id')
        return user_id, session
    else: 
        print('Failed to create_user')
        return None

def login_user(username, password):
    session = requests.Session()

    login_url = f'{BASE_URL}/auth/login'
    login_payload = {'username': username, 'password': password}
    response = session.post(login_url, data=login_payload, allow_redirects=False)

    if response.status_code == 200: 
        print(f"User '{username}' logged in successfully.")
        if token:
            session.headers.update({'Authorization': f'Bearer {token}'})
            return session  # ✅ Return session with access token in cookies
        else:
            print('Login failed (auth token)')
            return None
    else:
        print(f"Failed to log in user '{username}': {response.text}")
        return None

def get_user_id(session):
    url = f'{BASE_URL}/profile'
    response = session.get(url)

    if response.status_code == 200:
        user_data = response.json()
        user_id = user_data.get('_id')
        return user_id
    else:
        print('Failed to fecth user_id')

def add_ratings(user_id, session, isbns):
    url = f'{BASE_URL}/rating'

    print(f"🟡 Session Cookies Before Sending Ratings: {session.cookies.get_dict()}")

    for isbn in isbns:
        rating = random.randint(0,10)
        payload = {'user_id': user_id, 'isbn': isbn, 'rating': rating, 'comment': 'Auto-generated rating'}
        response = session.post(url, json=payload)
        if response.status_code != 200:
            print(f'Failed to add rating for book {isbn}: {response.text}')
        time.sleep(0.05)
        
async def test_recommendation(user_id):
    recommendations = await get_user_recommendations(user_id)

    if 'error' in recommendations:
        print(f'Recommendation error: {recommendations["error"]}')
    else:
        print("Top Recommended Books:")
        for rec in recommendations:
            print(f"{rec['title']} (ISBN: {rec['isbn']}) - Score: {rec.get('score', 'N/A')}")

async def get_isbns(n=50):
    isbns = books.find({}).limit(n)
    return [doc['isbn'] async for doc in isbns]

async def update_ratings(user_id, isbns, comment):
    tasks = []

    for isbn in isbns:

        rating = random.randint(0,10)
        task = ratings.update_one(
            {'user_id': user_id, 'isbn': isbn}, 
            {"$set": {'rating': rating, 'comment': comment, 'created_at': datetime.utcnow()}},
            upsert=True
        )
        tasks.append(task)
    results = await asyncio.gather(*tasks)

if __name__ == '__main__':

    username = 'test_user5'
    user_id = ObjectId('67ad83351e477145bac30007')
    comment = 'Auto-generated test rating'

    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    isbns = loop.run_until_complete(get_isbns())
    loop.run_until_complete(update_ratings(user_id, isbns, comment))
    """

    asyncio.run(test_recommendation(user_id))

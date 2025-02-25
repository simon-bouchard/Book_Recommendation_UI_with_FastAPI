import requests
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm 
import time 
import random
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.database import DATABASE_URL  # Use your DB connection

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
db = SessionLocal()

query = "SELECT isbn FROM books WHERE genre IS NULL OR genre = ''"
books_df = pd.read_sql(query, engine)

def fetch_genres(isbns, max_retries=5):
    base_url = "https://openlibrary.org/api/books"
    params = {"bibkeys": ",".join([f"ISBN:{isbn}" for isbn in isbns]), "format": "json", "jscmd": "data"}
    
    retries = 0
    while retries < max_retries:
        try:
            response = requests.get(base_url, params=params, timeout=15)  # Increase timeout
            response.raise_for_status()  # Raise error if status is not 200

            data = response.json()
            genre_mapping = {}

            for isbn_key, book_data in data.items():
                isbn = isbn_key.replace("ISBN:", "")
                subjects = book_data.get("subjects", [])
                genre = subjects[0]["name"] if subjects and isinstance(subjects[0], dict) else "Unknown"
                genre_mapping[isbn] = genre

            # Random delay to avoid rate limiting
            time.sleep(random.uniform(3, 8))  
            return genre_mapping

        except requests.exceptions.RequestException as e:
            print(f"⚠️ API Request Failed: {e}")
            retries += 1
            wait_time = min(4 ** retries + 2, 30)  # Exponential backoff (max 30s)
            print(f"🔄 Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

    print("❌ Max retries reached, skipping this batch.")
    return {}  # Return empty dict to prevent script failure


batch_size = 150
isbns_list = books_df["isbn"].tolist()

for i in tqdm(range(0, len(isbns_list), batch_size), desc="Fetching Genres"):
    batch = isbns_list[i : i + batch_size]
    genres = fetch_genres(batch)
    if not genres:
        print("⚠️ Skipping batch due to repeated API failures.")
        continue  # Skip batch if API is down

    update_queries = []
    for isbn, genre in genres.items():
        if isinstance(genre, list):  
            genre = genre[0]["name"] if isinstance(genre[0], dict) else genre[0]
        elif isinstance(genre, dict):  
            genre = genre.get("name", "Unknown")

        update_query = text("UPDATE books SET genre = :genre WHERE isbn = :isbn")
        update_queries.append({"genre": genre, "isbn": isbn})

    if update_queries:
        try:
            db.execute(update_query, update_queries)  # Bulk update
            db.commit()
            print(f"✅ Committed {len(update_queries)} updates.")
        except Exception as e:
            print(f"⚠️ Database Error: {e}")
            db.rollback()  # Prevent full failure

    time.sleep(1)  # Prevent API spam

db.close()
print("Genres updated successfully!")

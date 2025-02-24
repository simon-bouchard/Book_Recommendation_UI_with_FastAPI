import requests
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.database import DATABASE_URL  # Use your DB connection
from tqdm import tqdm  # For progress tracking

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
db = SessionLocal()

query = "SELECT isbn FROM books WHERE genre IS NULL OR genre = ''"
books_df = pd.read_sql(query, engine)

def fetch_genres(isbns):
    base_url = "https://openlibrary.org/api/books"
    params = {"bibkeys": ",".join([f"ISBN:{isbn}" for isbn in isbns]), "format": "json", "jscmd": "data"}
    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        data = response.json()
        genre_mapping = {}

        for isbn_key, book_data in data.items():
            isbn = isbn_key.replace("ISBN:", "")
            subjects = book_data.get("subjects", [])
            genre = subjects[0] if subjects else "Unknown"
            genre_mapping[isbn] = genre

        return genre_mapping
    return {}

batch_size = 100
isbns_list = books_df["isbn"].tolist()

for i in tqdm(range(0, len(isbns_list), batch_size), desc="Fetching Genres"):
    batch = isbns_list[i : i + batch_size]
    genres = fetch_genres(batch)

    for isbn, genre in genres.items():
        update_query = f"UPDATE books SET genre = '{genre}' WHERE isbn = '{isbn}'"
        db.execute(update_query)

    db.commit()

db.close()
print("Genres updated successfully!")

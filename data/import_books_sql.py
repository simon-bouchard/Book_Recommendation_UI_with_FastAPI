import pandas as pd
import numpy as np
import os
import sys
from dotenv import load_dotenv
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.database import DATABASE_URL
from app.models import Book

df = pd.read_csv('BX-Books.csv', encoding='ISO-8859-1', sep=';', quotechar='"', engine='python', on_bad_lines='skip')
df = df.replace({np.nan: None})

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
db = SessionLocal()

for _, row in df.iterrows():
    book = Book(
        isbn=row["ISBN"],
        title=row["Book-Title"],
        author=row["Book-Author"],
        year=int(row["Year-Of-Publication"]),
        publisher=row["Publisher"],
        image_url_s=row["Image-URL-S"],
        image_url_m=row["Image-URL-M"],
        image_url_l=row["Image-URL-L"],
    )
    db.merge(book)

db.commit()
db.close()

print("Books imported successfully!")

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.database import engine, Base
from app.models import Book, User, Rating

# Create all tables from models.py
Base.metadata.create_all(bind=engine)

print("Tables created successfully!")


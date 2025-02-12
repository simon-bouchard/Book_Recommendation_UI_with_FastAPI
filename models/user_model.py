import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

book_user_matrix = None
user_sparse_matrix = None
user_model = None
userid_to_index = None

load_dotenv()
client = AsyncIOMotorClient(os.getenv('MONGO_URI'))
db = client['book-recommendation']
books = db['Books']
ratings = db['ratings']
users = db['users']

async def reload_user_model():
    print('Model reload...')
    global book_user_matrix, user_sparse_matrix, user_model, userid_to_index

    ratings_cursor = ratings.find()
    ratings_list = await ratings_cursor.to_list(None)
    df_ratings = pd.DataFrame(ratings_list)

    if df_ratings.empty:
        return

    user_count = df_ratings['user_id'].value_counts()
    valid_users = user_count[user_count >= 20].index

    book_count = df_ratings['isbn'].value_counts()
    valid_books = book_count[book_count >= 100].index

    df_ratings = df_ratings[(df_ratings['user_id'].isin(valid_users)) & (df_ratings['isbn'].isin(valid_books))]

    df_ratings['rating'] = df_ratings['rating'].astype(int)

    book_user_matrix = df_ratings.pivot(index='user_id', columns='isbn', values='rating').fillna(0)
    user_sparse_matrix = csr_matrix(book_user_matrix.values)

    user_model = NearestNeighbors(metric='cosine', algorithm='brute')
    user_model.fit(user_sparse_matrix)

    userid_to_index = {user: idx for idx, user in enumerate(book_user_matrix.index)}

    print('Model reloaded')

async def get_user_recommendations(user: str, _id: bool = True):

    if _id:
        user_id = user
    else:
        user_entry = await users.find_one({'username': user})
        if not user_entry:
            return {'error': 'User not found'}
        user_id = user_entry.get('_id')

    if user_model is None or book_user_matrix is None or userid_to_index is None:
        await reload_user_model()
    
    if user_id not in userid_to_index:
        return { 'error': "User not found (users with less than 20 ratings can't get recommendations)"}

    query_index = userid_to_index[user_id]

    distances, indices = user_model.kneighbors(user_sparse_matrix[query_index], n_neighbors=50)
    similarities = 1 - distances

    similar_users = book_user_matrix.iloc[indices.flatten()]
    weighted_ratings = similar_users.multiply(similarities.T,axis=0)
    sum_similarity = similarities.sum(axis=1)
    
    if sum_similarity[0] == 0:
        return {'error': 'No recommendations could be generated.'}

    book_scores = weighted_ratings.sum() / sum_similarity[0]
    top_books = book_scores.nlargest(6)

    book_details = []
    for isbn, score in top_books.items():
        book = await books.find_one({'isbn': isbn})
        if book:
            book_details.append({'isbn': isbn, 'title': book['title'], 'score': score})

    return book_details


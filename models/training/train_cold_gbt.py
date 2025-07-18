import os
import sys
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict

from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from lightgbm import early_stopping, log_evaluation

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.database import SessionLocal
from app.table_models import Book, User, Interaction, BookSubject, Subject, UserFavSubject
from models.shared_utils import (
    load_attention_components,
    attention_pool,
    load_book_embeddings,
    get_item_idx_to_row,
    compute_subject_overlap,
    decompose_embeddings,
    PAD_IDX
)

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def fetch_data_from_sql():
    db = SessionLocal()
    try:
        print("🔄 Connecting to DB and loading interactions...")
        interactions = pd.read_sql("SELECT user_id, item_idx, rating FROM interactions WHERE rating IS NOT NULL", db.bind)
        users = pd.read_sql("SELECT * FROM users", db.bind)
        books = pd.read_sql("SELECT * FROM books", db.bind)

        print("📦 Loaded metadata from DB")

        user_fav = defaultdict(list)
        for row in db.query(UserFavSubject.user_id, UserFavSubject.subject_idx):
            user_fav[row.user_id].append(row.subject_idx)

        book_subj = defaultdict(list)
        for row in db.query(BookSubject.item_idx, BookSubject.subject_idx):
            book_subj[row.item_idx].append(row.subject_idx)

        db.close()
        return interactions, users, books, user_fav, book_subj

    except Exception as e:
        db.rollback()
        db.close()
        raise e

def main():
    interactions, users, books, user_fav, book_subj = fetch_data_from_sql()

    print("🧠 Loading subject attention components and book embeddings...")
    subject_emb, attn_weight, attn_bias = load_attention_components()
    book_embs, book_ids = load_book_embeddings()
    item_idx_to_row = get_item_idx_to_row(book_ids)
    book_emb_map = {item_idx: book_embs[item_idx_to_row[item_idx]] for item_idx in book_ids}

    rating_counts = interactions["user_id"].value_counts()
    interactions["is_warm"] = interactions["user_id"].map(lambda uid: rating_counts.get(uid, 0) >= 10)

    full_rows = []
    for row in interactions.itertuples(index=False):
        uid, iid, rating, is_warm = row
        if uid not in user_fav or iid not in book_subj:
            continue
        if iid not in book_emb_map:
            continue

        fav_subjs = user_fav[uid]
        book_subjs = book_subj[iid]

        user_emb = attention_pool([fav_subjs], subject_emb, attn_weight, attn_bias)[0]
        book_emb = book_emb_map[iid]
        subject_overlap = compute_subject_overlap(fav_subjs, book_subjs)

        full_rows.append({
            "user_id": uid,
            "item_idx": iid,
            "rating": rating,
            "is_warm": is_warm,
            "subject_overlap": subject_overlap,
            **decompose_embeddings(user_emb.unsqueeze(0), "user_emb"),
            **{f"book_emb_{i}": book_emb[i] for i in range(len(book_emb))}
        })

    df = pd.DataFrame(full_rows)
    df = df.merge(users, on="user_id", how="left")
    df = df.merge(
        books[["item_idx", "main_subject", "year", "filled_year", "language", "num_pages", "filled_num_pages"]],
        on="item_idx", how="left"
    )

    cat_cols = ["country", "filled_year", "filled_age", "main_subject", "year_bucket", "age_group", "language"]
    for col in cat_cols:
        df[col] = df[col].astype("category")

    cont_cols = ["age", "year", "num_pages", "subject_overlap"]
    emb_cols = [c for c in df.columns if c.startswith("user_emb_") or c.startswith("book_emb_")]
    features = cont_cols + cat_cols + emb_cols

    train = df[df["is_warm"] == True]
    val = df[df["is_warm"] == False]

    X_train = train[features].copy()
    X_val = val[features].copy()

    y_train = train["rating"]
    y_val = val["rating"]

    scaler = StandardScaler()
    X_train.loc[:, cont_cols] = scaler.fit_transform(X_train[cont_cols])
    X_val.loc[:, cont_cols] = scaler.transform(X_val[cont_cols])

    print("🚀 Training LightGBM model...")
    model = LGBMRegressor(
        objective="regression",
        metric="rmse",
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=42,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[early_stopping(50), log_evaluation(50)]
    )

    os.makedirs("models", exist_ok=True)
    with open("models/gbt_cold.pickle", "wb") as f:
        pickle.dump(model, f)

    print("✅ Saved: models/gbt_cold.pickle")

if __name__ == "__main__":
    main()

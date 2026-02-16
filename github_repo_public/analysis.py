import os
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from transformers import pipeline

# 1. Setup Data
if not os.path.exists("instagram.csv"):
    from synthetic_data_generator import generate_mock_data
    generate_mock_data()

insta, fb = pd.read_csv("instagram.csv"), pd.read_csv("facebook.csv")
df = pd.concat([insta.assign(platform='instagram'), fb.assign(platform='facebook')], ignore_index=True)
df = df.dropna(subset=["text"])

# 2. Clean & Embed
df["text_clean"] = df["text"].apply(lambda x: re.sub(r"http\S+|#\w+|\s+", " ", str(x)).strip())
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
embeddings = model.encode(df["text_clean"].tolist(), normalize_embeddings=True)

# 3. Clustering & Framing
kmeans = KMeans(n_clusters=min(5, len(df)), random_state=42, n_init="auto")
df["topic"] = kmeans.fit_predict(embeddings)

frame_labels = ["Road Safety", "Traffic Regulation", "Urban Order"]
frame_emb = model.encode(frame_labels, normalize_embeddings=True)
df["frame"] = [frame_labels[i] for i in (embeddings @ frame_emb.T).argmax(axis=1)]

# 4. Sentiment
sent_pipe = pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment", device=-1)
df["sentiment"] = [sent_pipe(t[:256])[0]['label'] for t in df["text_clean"]]

df.to_csv("final_results.csv", index=False)
print("Analysis complete. Results in final_results.csv")
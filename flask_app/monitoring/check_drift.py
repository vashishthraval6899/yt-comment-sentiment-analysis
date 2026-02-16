import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Config
THRESHOLD = 0.85   # lower = more drift
REFERENCE_PATH = "monitoring/reference_data.csv"
CURRENT_PATH = "monitoring/current_data.csv"

# Load data
ref = pd.read_csv(REFERENCE_PATH)["text"].astype(str)
cur = pd.read_csv(CURRENT_PATH)["text"].astype(str)

# Vectorize
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(pd.concat([ref, cur]))

n_ref = len(ref)
ref_vec = X[:n_ref].mean(axis=0).A1.reshape(1, -1)
cur_vec = X[n_ref:].mean(axis=0).A1.reshape(1, -1)

# Similarity
similarity = cosine_similarity(ref_vec, cur_vec)[0][0]

print(f"📊 Drift similarity score: {similarity:.4f}")

if similarity < THRESHOLD:
    print("⚠️ DRIFT DETECTED — retraining needed")
    exit(1)   # IMPORTANT
else:
    print("✅ No significant drift")
    exit(0)

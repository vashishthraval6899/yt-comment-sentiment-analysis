# app.py
import matplotlib
matplotlib.use('Agg')

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import os
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import joblib
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.dates as mdates


# Download required NLTK resources (safe for cloud)
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)
CORS(app)

plt.style.use('dark_background')

# -------------------------------
# Load Model + Vectorizer (Production Only)
# -------------------------------

MODEL_PATH = "flask_app/models/model.pkl"
VECTORIZER_PATH = "flask_app/models/vectorizer.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# -------------------------------
# Health Route
# -------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/")
def home():
    return "YouTube Comment Sentiment API is Running 🚀"


# -------------------------------
# Preprocessing
# -------------------------------

def preprocess_comment(comment):
    comment = comment.lower()
    comment = comment.strip()
    comment = re.sub(r'\n', ' ', comment)
    comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

    stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
    lemmatizer = WordNetLemmatizer()

    comment = ' '.join([
        lemmatizer.lemmatize(word)
        for word in comment.split()
        if word not in stop_words
    ])

    return comment


# -------------------------------
# Prediction Routes
# -------------------------------

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comments = data.get('comments')

    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        preprocessed = [preprocess_comment(c) for c in comments]
        transformed = vectorizer.transform(preprocessed)
        predictions = model.predict(transformed).tolist()
        predictions = [str(p) for p in predictions]

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    response = [{"comment": c, "sentiment": s} for c, s in zip(comments, predictions)]
    return jsonify(response)


@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    data = request.json
    comments_data = data.get('comments')

    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    try:
        comments = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]

        preprocessed = [preprocess_comment(c) for c in comments]
        transformed = vectorizer.transform(preprocessed)
        predictions = model.predict(transformed).tolist()
        predictions = [str(p) for p in predictions]

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    response = [
        {"comment": c, "sentiment": s, "timestamp": t}
        for c, s, t in zip(comments, predictions, timestamps)
    ]
    return jsonify(response)


# -------------------------------
# Plot Styling
# -------------------------------

def setup_plot_style():
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['grid.color'] = '#333333'
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.labelcolor'] = '#aaaaaa'
    plt.rcParams['xtick.color'] = '#aaaaaa'
    plt.rcParams['ytick.color'] = '#aaaaaa'


# -------------------------------
# Donut Sentiment Chart
# -------------------------------

@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        setup_plot_style()
        data = request.get_json()
        sentiment_counts = data.get('sentiment_counts')

        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]

        colors = ['#10b981', '#6b7280', '#ef4444']

        fig, ax = plt.subplots(figsize=(6, 6))
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            pctdistance=0.85,
            wedgeprops={'width': 0.4}
        )

        ax.axis('equal')

        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True, dpi=100)
        img_io.seek(0)
        plt.close()

        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------
# WordCloud
# -------------------------------

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get('comments')

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        preprocessed = [preprocess_comment(c) for c in comments]
        text = ' '.join(preprocessed)

        wc = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Blues',
            stopwords=set(stopwords.words('english')),
            collocations=False
        ).generate(text)

        img_io = io.BytesIO()
        wc.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------
# Trend Graph
# -------------------------------

@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        setup_plot_style()
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')

        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df['sentiment'] = df['sentiment'].astype(int)

        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)
        monthly_totals = monthly_counts.sum(axis=1)
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(monthly_percentages.index, monthly_percentages.get(1, 0),
                color='#10b981', linewidth=2, label='Positive')
        ax.fill_between(monthly_percentages.index,
                        monthly_percentages.get(1, 0),
                        color='#10b981', alpha=0.1)

        ax.plot(monthly_percentages.index, monthly_percentages.get(-1, 0),
                color='#ef4444', linewidth=2, label='Negative')
        ax.fill_between(monthly_percentages.index,
                        monthly_percentages.get(-1, 0),
                        color='#ef4444', alpha=0.1)

        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.legend(frameon=False)
        plt.tight_layout()

        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True, dpi=100)
        img_io.seek(0)
        plt.close()

        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------
# Run App
# -------------------------------

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
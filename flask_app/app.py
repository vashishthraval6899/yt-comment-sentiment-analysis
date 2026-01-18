# app.py
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import os
import sys

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import joblib
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import mlflow.sklearn
import matplotlib.dates as mdates



# ... other imports ...
from monitoring.log_production_data import log_texts

#Cloud machines do not have NLTK data.
nltk.download('stopwords')
nltk.download('wordnet')

IS_CI = os.getenv("CI", "false").lower() == "true"

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.before_request
def log_every_request():
    app.logger.warning(f"🔴 INCOMING REQUEST: {request.method} {request.path}")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

# Set global style for all charts to be Dark and Modern
plt.style.use('dark_background')

# Define the preprocessing function
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment

# Load the model and vectorizer
def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")

    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)

    client = MlflowClient()
    model_uri = f"models:/{model_name}/{model_version}"

    model = mlflow.sklearn.load_model(model_uri)
    vectorizer = joblib.load(vectorizer_path)

    return model, vectorizer

# Initialize the model and vectorizer
model = None
vectorizer = None  # Update paths and versions as needed

def get_model_and_vectorizer():
    global model, vectorizer

    if model is not None and vectorizer is not None:
        return model, vectorizer

    # 1️⃣ Try loading bundled model (PRODUCTION / RENDER)
    bundled_model_path = "flask_app/models/model.pkl"
    bundled_vectorizer_path = "flask_app/models/vectorizer.pkl"

    if os.path.exists(bundled_model_path) and os.path.exists(bundled_vectorizer_path):
        model = joblib.load(bundled_model_path)
        vectorizer = joblib.load(bundled_vectorizer_path)
        return model, vectorizer

    # 2️⃣ CI should never load models
    if IS_CI:
        raise RuntimeError("Model loading skipped in CI environment")

    # 3️⃣ Fallback to MLflow (LOCAL ONLY)
    model_name = "lgbm_model_V1"
    model_version = "1"
    vectorizer_path = "./tfidf_vectorizer_3000.pkl"

    model, vectorizer = load_model_and_vectorizer(
        model_name, model_version, vectorizer_path
    )

    return model, vectorizer


@app.route('/')
def home():
    return "Welcome to our flask api"

@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    data = request.json
    print("🔵 RAW REQUEST FROM CLIENT:", data)  # 👈 ADD THIS LINE

    comments_data = data.get('comments')
    
    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    try:
        comments = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]

        # Preprocess
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        log_texts(preprocessed_comments)
        
        # Transform
        model, vectorizer = get_model_and_vectorizer()
        transformed_comments = vectorizer.transform(preprocessed_comments)
        
        # Predict
        predictions = model.predict(transformed_comments).tolist()
        
        # Convert to string to ensure consistency ("Positive", "Negative", etc)
        predictions = [str(pred) for pred in predictions]
        
    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
    # Return string labels here. popup.js will convert them to numbers for math.
    response = [{"comment": c, "sentiment": s, "timestamp": t} for c, s, t in zip(comments, predictions, timestamps)]
    return jsonify(response)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comments = data.get('comments')
    
    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        log_texts(preprocessed_comments)
        
        model, vectorizer = get_model_and_vectorizer()
        transformed_comments = vectorizer.transform(preprocessed_comments)
        
        # CHANGE 3: Pass sparse matrix directly here as well
        predictions = model.predict(transformed_comments).tolist()
        
        predictions = [str(pred) for pred in predictions]
    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
    response = [{"comment": c, "sentiment": s} for c, s in zip(comments, predictions)]
    return jsonify(response)

# Helper to clear plot garbage and set basic aesthetics
def setup_plot_style():
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['grid.color'] = '#333333'
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.labelcolor'] = '#aaaaaa'
    plt.rcParams['xtick.color'] = '#aaaaaa'
    plt.rcParams['ytick.color'] = '#aaaaaa'

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
        
        # Modern Neon Colors
        colors = ['#10b981', '#6b7280', '#ef4444'] # Emerald, Gray, Red
        
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Create a Donut Chart (pie with a hole)
        wedges, texts, autotexts = ax.pie(
            sizes, 
            labels=labels, 
            colors=colors, 
            autopct='%1.1f%%', 
            startangle=140,
            pctdistance=0.85, # Move % towards edge
            wedgeprops={'width': 0.4, 'edgecolor': '#1e1e1e'} # width=0.4 makes it a donut
        )
        
        # Style text
        plt.setp(texts, size=12, weight="bold")
        plt.setp(autotexts, size=10, weight="bold", color="white")
        
        # Draw a circle in the center to ensure donut look (optional context, but pie handles it)
        ax.axis('equal')  

        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True, dpi=100)
        img_io.seek(0)
        plt.close()

        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_chart: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get('comments')

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        # Preprocess comments
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Combine all comments into a single string
        text = ' '.join(preprocessed_comments)

        # Generate the word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Blues',
            stopwords=set(stopwords.words('english')),
            collocations=False
        ).generate(text)

        # Save the word cloud to a BytesIO object
        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_wordcloud: {e}")
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500

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

        # Resample monthly percentages
        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)
        monthly_totals = monthly_counts.sum(axis=1)
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100
        
        for val in [-1, 0, 1]:
            if val not in monthly_percentages.columns:
                monthly_percentages[val] = 0
        
        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Use simple lines with fills (Area Chart effect)
        # Positive
        ax.plot(monthly_percentages.index, monthly_percentages[1], color='#10b981', linewidth=2, label='Positive')
        ax.fill_between(monthly_percentages.index, monthly_percentages[1], color='#10b981', alpha=0.1)
        
        # Negative
        ax.plot(monthly_percentages.index, monthly_percentages[-1], color='#ef4444', linewidth=2, label='Negative')
        ax.fill_between(monthly_percentages.index, monthly_percentages[-1], color='#ef4444', alpha=0.1)

        # Remove Neutral line to keep it clean, or keep it gray if you prefer
        # ax.plot(monthly_percentages.index, monthly_percentages[0], color='#4b5563', linestyle='--', linewidth=1, label='Neutral')

        # Clean up axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.xticks(rotation=0, ha='center')

        plt.legend(frameon=False, loc='upper left')
        plt.tight_layout()

        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True, dpi=100)
        img_io.seek(0)
        plt.close()

        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
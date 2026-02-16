import mlflow
import joblib
import os

MODEL_NAME = "lgbm_model_V1"
MODEL_VERSION = "1"

OUTPUT_DIR = "flask_app/models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Point to your LOCAL mlflow server
mlflow.set_tracking_uri("http://127.0.0.1:5001")

# Load model from MLflow
model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
model = mlflow.sklearn.load_model(model_uri)

# Save model
joblib.dump(model, f"{OUTPUT_DIR}/model.pkl")

# Save vectorizer (already local)
joblib.dump(
    joblib.load("tfidf_vectorizer_3000.pkl"),
    f"{OUTPUT_DIR}/vectorizer.pkl"
)

print("✅ Model and vectorizer exported successfully")

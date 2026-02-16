import subprocess

print("🔍 Checking data drift...")

result = subprocess.run(
    subprocess.run(["python", "scripts/mlflow_test.py"])
)

if result.returncode != 0:
    print("🚀 Drift detected → Retraining model")
    subprocess.run(["python", "scripts/train.py"])
else:
    print("🛑 Retraining skipped")

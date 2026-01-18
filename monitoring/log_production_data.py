import os
import pandas as pd
from datetime import datetime

LOG_PATH = "monitoring/current_data.csv"


def log_texts(texts: list[str]):
    print("🟢 DRIFT LOG | incoming texts:", texts)
    rows = []

    for t in texts:
        rows.append({
            "text": t,
            "timestamp": datetime.utcnow()
        })

    df = pd.DataFrame(rows)

    # Create file if not exists, else append
    if not os.path.exists(LOG_PATH):
        df.to_csv(LOG_PATH, index=False)
    else:
        df.to_csv(LOG_PATH, mode="a", header=False, index=False)

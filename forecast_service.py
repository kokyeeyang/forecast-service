from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import requests, os, json, logging
from datetime import datetime
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from logging.handlers import RotatingFileHandler

app = Flask(__name__)

# --- Directories ---
MODEL_DIR = "models"
LOG_DIR = "logs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

POINTER_FILE = os.path.join(MODEL_DIR, "latest.json")
LOG_FILE = os.path.join(LOG_DIR, "forecast_service.log")

# --- Logging setup (5MB max per file, keep 3 backups) ---
handler = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=3)
formatter = logging.Formatter(
    '%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
)
handler.setFormatter(formatter)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)


# --- Multi-feature LSTM ---
class MultiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
        super(MultiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


def fetch_metric(metric, datefrom, dateto, output):
    """Fetch one metric from PHP API and return DataFrame(period, total)."""
    url = "https://so-api.azurewebsites.net/ingress/ajax/api"
    params = {
        "metric": metric,
        "datefrom": datefrom,
        "dateto": dateto,
        "output": output
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, list):
        return pd.DataFrame(data)
    return pd.DataFrame(data.get("rows") or [])


@app.route("/forecast", methods=["GET"])
def forecast():
    # --- Parameters ---
    metric1 = request.args.get("metric1", "interviews")  # target
    horizon = int(request.args.get("horizon", 8))
    datefrom = request.args.get("datefrom", "2025-01-01")
    dateto = request.args.get("dateto", "2025-12-31")
    output = request.args.get("output", "weekly")
    train_flag = request.args.get("train", "false").lower() == "true"

    # Collect metrics (metric2, metric3, …)
    extras = []
    for k, v in request.args.items():
        if k.startswith("metric") and k != "metric1":
            extras.append(v)

    app.logger.info(f"Request received: metric1={metric1}, extras={extras}, "
                    f"horizon={horizon}, datefrom={datefrom}, dateto={dateto}, "
                    f"train={train_flag}")

    # --- Fetch data ---
    df_main = fetch_metric(metric1, datefrom, dateto, output)
    if df_main.empty or not {"period", "total"}.issubset(df_main.columns):
        app.logger.warning(f"Bad data for target metric={metric1}")
        return jsonify({"error": "Bad data for target", "metric": metric1})

    df_main["ds"] = pd.to_datetime(df_main["period"])
    df_main = df_main.sort_values("ds").reset_index(drop=True)
    df_main = df_main.rename(columns={"total": metric1})

    # Fetch extras
    for m in extras:
        df_m = fetch_metric(m, datefrom, dateto, output)
        if not df_m.empty and {"period", "total"}.issubset(df_m.columns):
            df_m["ds"] = pd.to_datetime(df_m["period"])
            df_m = df_m.rename(columns={"total": m})
            df_main = df_main.merge(df_m[["ds", m]], on="ds", how="left")

    df_main = df_main.fillna(0)

    # --- Prepare training data ---
    features = [metric1] + extras
    values = df_main[features].values.astype(float)
    target_idx = 0

    # Normalize
    mean, std = values.mean(axis=0), values.std(axis=0) + 1e-8
    values_norm = (values - mean) / std

    seq_len = 5
    X, y = [], []
    for i in range(len(values_norm) - seq_len):
        X.append(values_norm[i:i+seq_len])
        y.append(values_norm[i+seq_len, target_idx])
    if not X:
        app.logger.error("Not enough data for training")
        return jsonify({"error": "Not enough data for training"})

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # --- Model ---
    input_size = values.shape[1]
    model = MultiLSTM(input_size=input_size)

    # --- Try loading existing model if train=false ---
    latest_model = None
    if os.path.exists(POINTER_FILE) and not train_flag:
        with open(POINTER_FILE) as f:
            pointer = json.load(f)
        model_file = pointer.get("file")
        if model_file and os.path.exists(model_file):
            model.load_state_dict(torch.load(model_file))
            latest_model = model_file
            app.logger.info(f"Loaded existing model: {model_file}")

    # --- Training (if train=true or no model yet) ---
    mape, rmse = None, None
    if train_flag or latest_model is None:
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for epoch in range(50):
            model.train()
            optimizer.zero_grad()
            output_pred = model(X)
            loss = criterion(output_pred.squeeze(), y)
            loss.backward()
            optimizer.step()

        # Evaluate on last 4 weeks
        eval_size = min(4, len(y))
        if eval_size >= 2:
            y_true = y[-eval_size:].detach().numpy()
            with torch.no_grad():
                preds = model(X[-eval_size:]).squeeze().detach().numpy()
            y_true_denorm = y_true * std[target_idx] + mean[target_idx]
            preds_denorm = preds * std[target_idx] + mean[target_idx]
            mape = mean_absolute_percentage_error(y_true_denorm, preds_denorm)
            rmse = mean_squared_error(y_true_denorm, preds_denorm, squared=False)

        # Save model if error is acceptable
        if mape is None or mape < 0.2:  # 20% MAPE threshold
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            model_file = os.path.join(MODEL_DIR, f"{metric1}_{timestamp}.pth")
            torch.save(model.state_dict(), model_file)
            with open(POINTER_FILE, "w") as f:
                json.dump({"file": model_file, "time": timestamp}, f)
            app.logger.info(f"Model retrained and saved: {model_file} "
                            f"(MAPE={mape:.3f}, RMSE={rmse:.3f})")
        else:
            app.logger.warning(f"Model retrain skipped due to high error: "
                               f"MAPE={mape:.3f}, RMSE={rmse:.3f}")

    # --- Forecast ---
    model.eval()
    forecast_vals = []
    last_seq = values_norm[-seq_len:].copy()
    for _ in range(horizon):
        seq_tensor = torch.tensor([last_seq], dtype=torch.float32)
        with torch.no_grad():
            pred_norm = model(seq_tensor).item()
        pred = pred_norm * std[target_idx] + mean[target_idx]
        forecast_vals.append(pred)
        last_seq = np.vstack([last_seq[1:], [[pred_norm] + [0]*(input_size-1)]])

    future_dates = pd.date_range(df_main["ds"].iloc[-1], periods=horizon+1, freq="W")[1:]
    forecast_data = [
        {"period": str(date.date()), f"forecast_{metric1}": round(val, 2)}
        for date, val in zip(future_dates, forecast_vals)
    ]

    return jsonify({
        "metrics": {"target": metric1, "extras": extras},
        "history_points": len(df_main),
        "history_sample": df_main.head(5).to_dict(orient="records"),
        "forecast": forecast_data,
        "evaluation": {"mape": mape, "rmse": rmse}
    })

if __name__ == "__main__":
    import os
    dev_mode = os.environ.get("DEV", "false").lower() == "true"
    port = int(os.environ.get("PORT", 5001 if dev_mode else 5000))
    
    if dev_mode:
        print(f"🚀 Running in DEV mode on http://localhost:{port}")
        app.run(host="0.0.0.0", port=port, debug=True)
    else:
        print("⚠️  Running in production mode - start with Gunicorn instead of Flask dev server")
from flask import Flask, request, jsonify, send_file
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import requests, os, json, logging, sys, zipfile, io
from datetime import datetime
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from logging.handlers import RotatingFileHandler

app = Flask(__name__)

# ==========================================================
# 1️⃣ Directories and Logging Setup
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Local vs Production directory setup
if os.environ.get("DEV", "false").lower() == "true":
    DATA_DIR = BASE_DIR  # local: next to forecast_service.py
else:
    DATA_DIR = "/var/data"  # Render: persistent disk

MODEL_DIR = os.path.join(DATA_DIR, "models")
LOG_DIR = os.path.join(DATA_DIR, "logs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

POINTER_FILE = os.path.join(MODEL_DIR, "latest.json")
LOG_FILE = os.path.join(LOG_DIR, "forecast_service.log")

# Logging setup (console + file)
handler = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=3)
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
handler.setFormatter(formatter)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
app.logger.addHandler(console_handler)

print(f"📂 MODEL_DIR = {MODEL_DIR}", flush=True)
print(f"📂 LOG_DIR = {LOG_DIR}", flush=True)


# ==========================================================
# 2️⃣ Model Definition
# ==========================================================
class MultiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
        super(MultiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ==========================================================
# 3️⃣ GitHub Artifact Fetch Helper
# ==========================================================
def ensure_model_available():
    """Ensure a model exists locally; if not, fetch latest artifact from GitHub Actions."""
    if os.path.exists(POINTER_FILE):
        app.logger.info("✅ Model pointer found locally.")
        return True

    github_token = os.environ.get("GITHUB_TOKEN")
    github_repo = os.environ.get("GITHUB_REPO")

    if not github_token or not github_repo:
        app.logger.warning("⚠️  Missing GitHub credentials. Cannot auto-fetch model.")
        return False

    try:
        headers = {"Authorization": f"Bearer {github_token}"}
        runs_url = f"https://api.github.com/repos/{github_repo}/actions/runs"
        runs_resp = requests.get(runs_url, headers=headers)
        runs_resp.raise_for_status()
        runs = runs_resp.json().get("workflow_runs", [])
        if not runs:
            app.logger.error("❌ No workflow runs found.")
            return False

        latest_run_id = runs[0]["id"]
        artifacts_url = f"https://api.github.com/repos/{github_repo}/actions/runs/{latest_run_id}/artifacts"
        artifacts_resp = requests.get(artifacts_url, headers=headers)
        artifacts_resp.raise_for_status()
        artifacts = artifacts_resp.json().get("artifacts", [])
        if not artifacts:
            app.logger.error("❌ No artifacts in latest run.")
            return False

        artifact_id = artifacts[0]["id"]
        download_url = f"https://api.github.com/repos/{github_repo}/actions/artifacts/{artifact_id}/zip"
        download_resp = requests.get(download_url, headers=headers)
        download_resp.raise_for_status()

        # Extract to model dir
        with zipfile.ZipFile(io.BytesIO(download_resp.content)) as z:
            z.extractall(MODEL_DIR)

        app.logger.info("✅ Model artifact downloaded & extracted from GitHub.")
        return True
    except Exception as e:
        app.logger.error(f"❌ Failed to fetch GitHub model: {e}")
        return False


# ==========================================================
# 4️⃣ PHP API Fetcher
# ==========================================================
def fetch_metric(metric, datefrom, dateto, output):
    """Fetch metric from PHP API."""
    url = "https://so-api.azurewebsites.net/ingress/ajax/api"
    params = {
        "metric": metric,
        "datefrom": datefrom,
        "dateto": dateto,
        "output": output,
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, list):
        return pd.DataFrame(data)
    return pd.DataFrame(data.get("rows") or [])


# ==========================================================
# 5️⃣ Forecast Endpoint
# ==========================================================
@app.route("/forecast", methods=["GET"])
def forecast():
    metric1 = request.args.get("metric1", "interviews")
    horizon = int(request.args.get("horizon", 8))
    datefrom = request.args.get("datefrom", "2025-01-01")
    dateto = request.args.get("dateto", "2025-12-31")
    output = request.args.get("output", "weekly")
    train_flag = request.args.get("train", "false").lower() == "true"
    extras = [v for k, v in request.args.items() if k.startswith("metric") and k != "metric1"]

    app.logger.info(f"🔹 Request: {metric1}, extras={extras}, horizon={horizon}, train={train_flag}")

    # --- Fetch data ---
    df_main = fetch_metric(metric1, datefrom, dateto, output)
    if df_main.empty or not {"period", "total"}.issubset(df_main.columns):
        return jsonify({"error": "Bad data for target metric", "metric": metric1}), 400

    df_main["ds"] = pd.to_datetime(df_main["period"])
    df_main = df_main.sort_values("ds").reset_index(drop=True)
    df_main = df_main.rename(columns={"total": metric1})

    for m in extras:
        df_m = fetch_metric(m, datefrom, dateto, output)
        if not df_m.empty and {"period", "total"}.issubset(df_m.columns):
            df_m["ds"] = pd.to_datetime(df_m["period"])
            df_m = df_m.rename(columns={"total": m})
            df_main = df_main.merge(df_m[["ds", m]], on="ds", how="left")

    df_main = df_main.fillna(0)

    # --- Training data ---
    features = [metric1] + extras
    values = df_main[features].values.astype(float)
    target_idx = 0
    mean, std = values.mean(axis=0), values.std(axis=0) + 1e-8
    values_norm = (values - mean) / std

    seq_len = 5
    X, y = [], []
    for i in range(len(values_norm) - seq_len):
        X.append(values_norm[i:i+seq_len])
        y.append(values_norm[i+seq_len, target_idx])

    if not X:
        return jsonify({"error": "Not enough data for training"}), 400

    X = torch.tensor(np.array(X, dtype=np.float32))
    y = torch.tensor(np.array(y, dtype=np.float32))
    model = MultiLSTM(input_size=values.shape[1])

    # --- Try loading existing model ---
    latest_model = None
    if os.path.exists(POINTER_FILE) and not train_flag:
        with open(POINTER_FILE) as f:
            pointer = json.load(f)
        model_file = pointer.get("file")
        if model_file and os.path.exists(model_file):
            model.load_state_dict(torch.load(model_file))
            latest_model = model_file
            app.logger.info(f"✅ Loaded model: {model_file}")

    # --- Train / retrain ---
    mape, rmse = None, None
    if train_flag or latest_model is None:
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for epoch in range(50):
            model.train()
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out.squeeze(), y)
            loss.backward()
            optimizer.step()

        eval_size = min(4, len(y))
        if eval_size >= 2:
            y_true = y[-eval_size:].numpy()
            preds = model(X[-eval_size:]).squeeze().detach().numpy()
            y_true_denorm = y_true * std[target_idx] + mean[target_idx]
            preds_denorm = preds * std[target_idx] + mean[target_idx]
            mape = mean_absolute_percentage_error(y_true_denorm, preds_denorm)
            rmse = np.sqrt(mean_squared_error(y_true_denorm, preds_denorm))

        # Save model if acceptable
        if mape is None or mape < 0.2:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            model_file = os.path.join(MODEL_DIR, f"{metric1}_{timestamp}.pth")
            torch.save(model.state_dict(), model_file)
            with open(POINTER_FILE, "w") as f:
                json.dump({"file": model_file, "time": timestamp, "mape": mape, "rmse": rmse}, f)

            # 🧹 Keep only 2 newest
            all_models = sorted(
                [f for f in os.listdir(MODEL_DIR) if f.endswith(".pth")],
                key=lambda f: os.path.getmtime(os.path.join(MODEL_DIR, f)),
                reverse=True,
            )
            for old in all_models[2:]:
                os.remove(os.path.join(MODEL_DIR, old))

            app.logger.info(f"💾 Saved model {model_file} (MAPE={mape:.3f}, RMSE={rmse:.3f})")
        else:
            app.logger.warning(f"⚠️ Retrain skipped: high error MAPE={mape}, RMSE={rmse}")

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
        last_seq = np.vstack([last_seq[1:], [[pred_norm] + [0]*(values.shape[1]-1)]])

    future_dates = pd.date_range(df_main["ds"].iloc[-1], periods=horizon+1, freq="W")[1:]
    forecast_data = [
        {"period": str(date.date()), f"forecast_{metric1}": round(val, 2)}
        for date, val in zip(future_dates, forecast_vals)
    ]

    return jsonify({
        "metrics": {"target": metric1, "extras": extras},
        "history_points": len(df_main),
        "forecast": forecast_data,
        "evaluation": {"mape": mape, "rmse": rmse},
    })


# ==========================================================
# 6️⃣ Model Management Endpoints
# ==========================================================
@app.route("/download_model", methods=["GET"])
def download_model():
    """Download latest trained model."""
    if not os.path.exists(POINTER_FILE):
        return jsonify({"error": "No model pointer file found"}), 404
    with open(POINTER_FILE) as f:
        pointer = json.load(f)
    model_file = pointer.get("file")
    if not model_file or not os.path.exists(model_file):
        return jsonify({"error": "Model file not found"}), 404
    return send_file(model_file, as_attachment=True)


@app.route("/refresh_model", methods=["POST"])
def refresh_model():
    """Manually trigger GitHub model fetch."""
    success = ensure_model_available()
    if success:
        return jsonify({"status": "success", "message": "Model refreshed from GitHub"}), 200
    else:
        return jsonify({"status": "error", "message": "Failed to refresh model from GitHub"}), 500


# ==========================================================
# 7️⃣ Entry Point
# ==========================================================
if __name__ == "__main__":
    dev_mode = os.environ.get("DEV", "false").lower() == "true"
    if not dev_mode:
        ensure_model_available()

    port = int(os.environ.get("PORT", 5001 if dev_mode else 5000))
    if dev_mode:
        print(f"🚀 DEV mode at http://localhost:{port}")
        app.run(host="0.0.0.0", port=port, debug=True)
    else:
        print("⚠️ PROD mode — use Gunicorn in Docker/Render")
        app.run(host="0.0.0.0", port=port)

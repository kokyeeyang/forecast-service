# ---------------------------
# 1. Use official Python image
# ---------------------------
FROM python:3.10-slim

# ---------------------------
# 2. Set working directory
# ---------------------------
WORKDIR /app

# ---------------------------
# 3. Install dependencies
# ---------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------
# 4. Copy app code
# ---------------------------
COPY . .

# ---------------------------
# 5. Expose default port
# ---------------------------
EXPOSE 5000

# ---------------------------
# 6. Start Gunicorn in Render (production)
# ---------------------------
# Note: Locally you can still run:
#   DEV=true python forecast_service.py
# which bypasses this CMD and uses Flask dev server
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "forecast_service:app"]

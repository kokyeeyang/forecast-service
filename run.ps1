param (
    [Parameter(Mandatory = $true)]
    [ValidateSet("dev", "prod")]
    [string]$mode
)

Write-Host "🔹 Activating virtual environment..."
. .\venv\Scripts\Activate

if ($mode -eq "dev") {
    Write-Host "🚀 Starting forecast_service.py in DEVELOPMENT mode..."
    $env:DEV="true"
    python forecast_service.py
}
elseif ($mode -eq "prod") {
    Write-Host "⚙️  Starting forecast_service.py in PRODUCTION mode (Gunicorn)..."
    $env:DEV="false"
    gunicorn -w 4 -b 0.0.0.0:5000 forecast_service:app
}

# Host API for the challenge (set PORT if your host injects it)
$port = if ($env:PORT) { $env:PORT } else { "8080" }
Set-Location $PSScriptRoot
# Single process keeps RAM low on 512MB plans (avoids N×model copies).
python -m uvicorn main:app --host 0.0.0.0 --port $port --workers 1

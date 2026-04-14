@echo off
setlocal

python scripts\run_pipeline.py --config configs\config.yaml
if %errorlevel% neq 0 (
  echo Pipeline failed.
  exit /b %errorlevel%
)

echo Pipeline completed successfully.

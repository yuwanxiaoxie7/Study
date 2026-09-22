@echo off
setlocal
cd /d "%~dp0"
python scripts\package_cloud.py
if errorlevel 1 (
  echo Packaging failed. Check Python and project files.
  pause
  exit /b 1
)
echo Upload uav_segmentation_dino_4090_cloud.zip and extract it on the server.
echo Run: bash run_all.sh --data-root /path/to/dataset
pause

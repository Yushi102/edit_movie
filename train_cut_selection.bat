@echo off
chcp 65001 > nul
echo Cut Selection Model Training
echo.

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Set PYTHONPATH
set PYTHONPATH=%CD%

REM Install japanize-matplotlib if needed
echo Checking dependencies...
pip install japanize-matplotlib -q

echo.
echo Starting training...
echo Visualization will be saved to checkpoints_cut_selection/training_progress.png
echo.

REM Run training
python src\cut_selection\train_cut_selection.py --config configs\config_cut_selection.yaml

echo.
echo Complete!
echo.
echo Results:
echo   - Model: checkpoints_cut_selection\best_model.pth
echo   - Graph: checkpoints_cut_selection\training_progress.png
echo   - History: checkpoints_cut_selection\training_history.csv
echo.
pause

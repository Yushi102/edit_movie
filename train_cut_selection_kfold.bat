@echo off
chcp 65001 > nul
echo K-Fold Cross Validation for Cut Selection Model
echo.

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Set PYTHONPATH
set PYTHONPATH=%CD%

REM Install japanize-matplotlib if needed
echo Checking dependencies...
pip install japanize-matplotlib scikit-learn -q

echo.
echo Step 1: Combining train and val data...
python scripts\create_combined_data_for_kfold.py

if errorlevel 1 (
    echo.
    echo ❌ Failed to combine data
    pause
    exit /b 1
)

echo.
echo Step 2: Starting K-Fold Cross Validation...
echo This will train %n_folds% models (one per fold)
echo Results will be saved to checkpoints_cut_selection_kfold/
echo.

REM Run K-Fold training
python src\cut_selection\train_cut_selection_kfold.py --config configs\config_cut_selection_kfold.yaml

echo.
echo Complete!
echo.
echo Results:
echo   - Models: checkpoints_cut_selection_kfold\fold_*_best_model.pth
echo   - Comparison: checkpoints_cut_selection_kfold\kfold_comparison.png
echo   - Summary: checkpoints_cut_selection_kfold\kfold_summary.csv
echo.
pause

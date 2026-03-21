@echo off
echo ========================================
echo One-Button Training Pipeline
echo Feature Extraction + Training
echo ========================================
echo.
echo This will execute the complete training pipeline:
echo   1. Feature Extraction (5-10 min per video)
echo   2. Label Extraction (few seconds)
echo   3. Temporal Features Addition (few minutes)
echo   4. Dataset Creation (few minutes)
echo   5. Model Training (1-2 hours)
echo.
echo Press Ctrl+C to cancel, or
pause

python scripts/train_pipeline_onebutton.py --config configs/config_cut_selection_fullvideo.yaml

echo.
echo ========================================
echo Pipeline execution completed
echo ========================================
pause

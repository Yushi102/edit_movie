@echo off
echo ========================================
echo One-Button Training Pipeline
echo Feature Extraction + Training
echo WITH AUDIO SEPARATION
echo ========================================
echo.
echo This will execute the complete training pipeline with audio separation:
echo   1. Feature Extraction (5-10 min per video)
echo      - Audio separation enabled (improves Whisper accuracy)
echo   2. Label Extraction (few seconds)
echo   3. Temporal Features Addition (few minutes)
echo   4. Dataset Creation (few minutes)
echo   5. Model Training (1-2 hours)
echo.
echo Audio Separation:
echo   - Separates game audio from voice
echo   - Improves Whisper transcription accuracy by 10-30%%
echo   - Quality: balanced (3-5 min per 10-min video)
echo.
echo Press Ctrl+C to cancel, or
pause

python scripts/train_pipeline_onebutton.py --config configs/config_cut_selection_fullvideo.yaml --enable-audio-separation

echo.
echo ========================================
echo Pipeline execution completed
echo ========================================
pause

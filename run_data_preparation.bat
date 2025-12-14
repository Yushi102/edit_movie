@echo off
REM 動画編集AI - データ準備スクリプト

REM Pythonパスを設定
set PYTHONPATH=%PYTHONPATH%;%CD%\src

echo ================================================================================
echo 動画編集AI - データ準備
echo ================================================================================
echo.
echo このスクリプトは以下の処理を実行します:
echo   1. XMLからラベル抽出
echo   2. 動画から特徴量抽出
echo   3. データ統合
echo.
echo ================================================================================
echo.

echo [1/3] XMLからラベル抽出中...
python src\data_preparation\premiere_xml_parser.py

if errorlevel 1 (
    echo.
    echo エラー: XMLパースに失敗しました
    exit /b 1
)

echo.
echo [2/3] 動画から特徴量抽出中（時間がかかります）...
python src\data_preparation\extract_video_features_parallel.py

if errorlevel 1 (
    echo.
    echo エラー: 特徴量抽出に失敗しました
    exit /b 1
)

echo.
echo [3/3] データ統合中...
python src\data_preparation\data_preprocessing.py

if errorlevel 1 (
    echo.
    echo エラー: データ統合に失敗しました
    exit /b 1
)

echo.
echo ================================================================================
echo データ準備完了！
echo ================================================================================
echo.
echo 学習用データセット: data\processed\master_training_data.csv
echo.
echo 次は学習を実行してください:
echo   run_training.bat
echo.

exit /b 0

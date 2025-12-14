@echo off
REM 動画編集AI - 推論実行スクリプト

REM Pythonパスを設定
set PYTHONPATH=%PYTHONPATH%;%CD%\src

REM 引数チェック
if "%~1"=="" (
    echo Usage: run_inference.bat ^<video_path^> [model_path] [output_path]
    echo.
    echo Example:
    echo   run_inference.bat "D:\videos\my_video.mp4"
    echo   run_inference.bat "D:\videos\my_video.mp4" "models\checkpoints_50epochs\best_model.pth"
    exit /b 1
)

REM デフォルト値を設定
set VIDEO_PATH=%~1
set MODEL_PATH=%~2
set OUTPUT_PATH=%~3

if "%MODEL_PATH%"=="" set MODEL_PATH=models\checkpoints_50epochs\best_model.pth
if "%OUTPUT_PATH%"=="" set OUTPUT_PATH=outputs\inference_results\result.xml

echo ================================================================================
echo 動画編集AI - 推論実行
echo ================================================================================
echo.
echo 入力動画: %VIDEO_PATH%
echo モデル: %MODEL_PATH%
echo 出力XML: %OUTPUT_PATH%
echo.
echo ================================================================================
echo.

REM 一時ファイル名
set TEMP_XML=outputs\inference_results\temp_%RANDOM%.xml

echo [1/3] 推論を実行中...
python src\inference\inference_pipeline.py "%VIDEO_PATH%" --model "%MODEL_PATH%" --output "%TEMP_XML%"

if errorlevel 1 (
    echo.
    echo エラー: 推論に失敗しました
    exit /b 1
)

echo.
echo [2/3] テロップをグラフィックに変換中...
python src\inference\fix_telop_simple.py "%TEMP_XML%" "%OUTPUT_PATH%"

if errorlevel 1 (
    echo.
    echo エラー: テロップ変換に失敗しました
    exit /b 1
)

echo.
echo [3/3] 完了！
echo.
echo ================================================================================
echo 出力XMLファイル: %OUTPUT_PATH%
echo ================================================================================
echo.
echo Premiere Proで上記のXMLファイルを開いてください。
echo.

REM 一時ファイルを削除
if exist "%TEMP_XML%" del "%TEMP_XML%"

exit /b 0

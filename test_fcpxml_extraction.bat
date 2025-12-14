@echo off
REM FCPXML抽出テストスクリプト
REM 使用方法: test_fcpxml_extraction.bat "C:\Users\yushi\Documents\プログラム\editxml\your_file.fcpxml"

echo ========================================
echo FCPXML 抽出テスト
echo ========================================
echo.

if "%~1"=="" (
    echo エラー: FCPXMLファイルのパスを指定してください
    echo 使用例: test_fcpxml_extraction.bat "C:\Users\yushi\Documents\プログラム\editxml\your_file.fcpxml"
    exit /b 1
)

set FCPXML_FILE=%~1
set OUTPUT_DIR=fcpxml_test_output

echo 入力ファイル: %FCPXML_FILE%
echo 出力ディレクトリ: %OUTPUT_DIR%
echo.

REM 出力ディレクトリを作成
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM Pythonスクリプトを実行
echo パース開始...
python fcpxml_to_tracks.py "%FCPXML_FILE%" --output "%OUTPUT_DIR%" --format both --fps 10.0 --max-tracks 20

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo 抽出完了！
    echo ========================================
    echo.
    echo 出力ファイル:
    dir /b "%OUTPUT_DIR%"
    echo.
    echo CSVファイルを確認してください:
    echo %OUTPUT_DIR%\*_tracks.csv
) else (
    echo.
    echo エラーが発生しました。
)

pause

@echo off
REM 複数のFCPXMLファイルを一括処理
REM 使用方法: batch_test_fcpxml.bat "path\to\fcpxml_directory"

echo ========================================
echo FCPXML 一括抽出テスト
echo ========================================
echo.

if "%~1"=="" (
    echo エラー: FCPXMLファイルが含まれるディレクトリを指定してください
    echo 使用例: batch_test_fcpxml.bat "path\to\fcpxml_directory"
    exit /b 1
)

set INPUT_DIR=%~1
set OUTPUT_DIR=fcpxml_batch_output

echo 入力ディレクトリ: %INPUT_DIR%
echo 出力ディレクトリ: %OUTPUT_DIR%
echo.

REM 出力ディレクトリを作成
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM すべてのFCPXMLファイルを処理
set COUNT=0
for %%F in ("%INPUT_DIR%\*.fcpxml") do (
    echo.
    echo ----------------------------------------
    echo 処理中: %%~nxF
    echo ----------------------------------------
    python fcpxml_to_tracks.py "%%F" --output "%OUTPUT_DIR%" --format both --fps 10.0 --max-tracks 20
    set /a COUNT+=1
)

echo.
echo ========================================
echo 一括処理完了！
echo ========================================
echo 処理ファイル数: %COUNT%
echo.
echo 出力ファイル:
dir /b "%OUTPUT_DIR%"
echo.
echo CSVファイルを確認してください:
echo %OUTPUT_DIR%\*_tracks.csv

pause

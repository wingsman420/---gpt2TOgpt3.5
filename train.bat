@echo off
cd /d "%~dp0"

:: Switch console to UTF-8 so Chinese output from Python renders correctly.
:: IMPORTANT: keep this file pure ASCII. cmd parses .bat with the system
:: codepage; Chinese characters here corrupt the line structure.
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

echo ============================================================
echo   GPT-2 SFT Training
echo ============================================================
echo.
echo   Config : src\_sft\config.py      ^<-- edit this
echo   Data   : data\processed\train.jsonl
echo   Output : checkpoints\sft.pt
echo.
echo   After editing the config, just run this file again.
echo ============================================================
echo.

.venv\Scripts\python.exe src\train_sft.py %*

echo.
echo ------------------------------------------------------------
echo   Finished (exit code %ERRORLEVEL%)
echo ------------------------------------------------------------
pause

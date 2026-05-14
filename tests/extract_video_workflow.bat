@echo off
setlocal

if "%~1"=="" (
    echo Usage: Drag and drop a video file onto this BAT.
    echo.
    echo This will create a JSON file next to the video.
    pause
    exit /b 1
)

set "VIDEO=%~1"
set "OUT=%~dp1%~n1.json"
set "SCRIPT_DIR=%~dp0"
set "SCRIPT=%SCRIPT_DIR%extract_video_workflow.py"

if not exist "%SCRIPT%" (
    echo Error: Python script not found:
    echo %SCRIPT%
    pause
    exit /b 2
)

echo Extracting workflow...
echo Input : %VIDEO%
echo Output: %OUT%
echo.

py "%SCRIPT%" "%VIDEO%" --out "%OUT%"
if errorlevel 1 (
    echo.
    echo Extraction failed.
    pause
    exit /b 3
)

echo.
echo Done. Workflow saved to:
echo %OUT%
pause
exit /b 0

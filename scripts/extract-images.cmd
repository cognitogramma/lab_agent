@echo off
REM Launcher for extract_pdf_images.py using the project venv, so you never
REM have to type the interpreter path or activate anything.
REM
REM   extract-images "C:\papers"
REM   extract-images "C:\papers" --mode figures --dpi 600
REM
REM You can also drag a folder or a PDF onto this file in Explorer.

setlocal
set "HERE=%~dp0"
set "PY=%HERE%..\.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo Could not find the project venv at "%PY%".
    echo Create it, then: .venv\Scripts\python.exe -m pip install pymupdf numpy
    exit /b 1
)

if "%~1"=="" (
    echo Usage: extract-images ^<file.pdf or folder^> [options]
    echo Run with --help for the full option list.
    exit /b 1
)

"%PY%" "%HERE%extract_pdf_images.py" %*
set "RC=%ERRORLEVEL%"

REM Drag-and-drop from Explorer spawns "cmd /c ...", and the window would
REM otherwise close before the summary could be read. But PowerShell also
REM invokes .cmd files with /c, so that test alone would stall a scripted
REM run -- hence the opt-out.
if defined EXTRACT_IMAGES_NOPAUSE goto :done
echo %CMDCMDLINE% | find /i "/c" >nul
if not errorlevel 1 pause

:done
exit /b %RC%

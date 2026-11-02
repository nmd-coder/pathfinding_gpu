@echo off
REM Kich ban quay video demo: chay tron bo + ve bieu do
REM Chay tu thu muc goc du an:  scripts\run_demo.bat
cd /d "%~dp0\.."
if not exist pathfinder.exe (echo Build first: build.bat & exit /b 1)
if not exist results mkdir results

pathfinder.exe --mode demo --queries 3
if errorlevel 1 echo [WARN] demo returned a non-zero exit code

where python >nul 2>nul && python scripts\plot_results.py results
echo.
echo Results: results\*.csv   Figures: results\figures\*.png

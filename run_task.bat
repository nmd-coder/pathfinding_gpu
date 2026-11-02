@echo off
REM ===========================================================================
REM  run_task.bat - Chay lenh trong logs\task.cmd, hien thi truc tiep va ghi log
REM  vao logs\task_out.txt (dung de Claude dieu khien build/chay thuc nghiem).
REM ===========================================================================
cd /d "%~dp0"
if not exist logs mkdir logs
del /q logs\task_done.txt 2>nul
title pathfinding_gpu task runner
powershell -NoProfile -ExecutionPolicy Bypass -Command "$OutputEncoding=[Console]::OutputEncoding=[Text.Encoding]::UTF8; & cmd /c 'logs\task.cmd' 2>&1 | Tee-Object -FilePath 'logs\task_out.txt'"
echo finished> logs\task_done.txt
echo.
echo ===== TASK FINISHED - you can close this window =====
timeout /t 15 >nul

@echo off
REM ===========================================================================
REM  Bien dich tren Windows bang nvcc + MSVC
REM  YEU CAU: chay trong "x64 Native Tools Command Prompt for VS 2019/2022"
REM           (hoac da chay vcvars64.bat) va da cai CUDA Toolkit 11.6+
REM  Doi ARCH neu nvcc bao khong ho tro "native", vi du:
REM     GTX 10xx: sm_61 | RTX 20xx: sm_75 | RTX 30xx: sm_86 | RTX 40xx: sm_89 | RTX 50xx: sm_120
REM ===========================================================================
setlocal
if "%ARCH%"=="" set ARCH=-arch=native

where nvcc >nul 2>nul || (echo [ERROR] nvcc not found. Install CUDA Toolkit and reopen the terminal. & exit /b 1)
where cl   >nul 2>nul || (echo [ERROR] cl.exe not found. Open "x64 Native Tools Command Prompt for VS". & exit /b 1)

echo Building pathfinder.exe with %ARCH% ...
nvcc -O3 -std=c++17 %ARCH% -lineinfo -Iinclude -Isrc ^
  -Xcompiler "/O2 /openmp /utf-8 /EHsc /D_CRT_SECURE_NO_WARNINGS /DNOMINMAX" ^
  src\main.cpp src\graph.cpp src\cpu_sssp.cpp src\verify.cpp src\report.cpp src\term.cpp ^
  src\gpu_device.cu src\gpu_sssp.cu ^
  -o pathfinder.exe
if errorlevel 1 (echo [ERROR] Build failed. & exit /b 1)
echo [OK] Built pathfinder.exe
echo Run:  pathfinder.exe --mode demo
endlocal

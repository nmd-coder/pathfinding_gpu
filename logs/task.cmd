@echo off
echo ==== GPU adapters
powershell -NoProfile -Command "Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion, AdapterRAM, VideoProcessor | Format-List"
echo ==== NVIDIA folders
dir /b "C:\Program Files\NVIDIA Corporation" 2>nul
dir /b "C:\Windows\System32\nvidia-smi.exe" 2>nul
if exist "C:\Windows\System32\nvidia-smi.exe" "C:\Windows\System32\nvidia-smi.exe"
echo ==== CPU / RAM / OS
powershell -NoProfile -Command "Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors | Format-List; Get-CimInstance Win32_ComputerSystem | Select-Object @{n='RAM_GB';e={[math]::Round($_.TotalPhysicalMemory/1GB,1)}} | Format-List; (Get-CimInstance Win32_OperatingSystem).Caption"
echo ==== Disk free
powershell -NoProfile -Command "Get-PSDrive -PSProvider FileSystem | Select-Object Name, @{n='FreeGB';e={[math]::Round($_.Free/1GB,1)}} | Format-Table -AutoSize"
echo ==== Visual Studio installs
dir /b "C:\Program Files\Microsoft Visual Studio" 2>nul
dir /b "C:\Program Files (x86)\Microsoft Visual Studio" 2>nul
echo ==== winget
where winget
echo TASK_FINISHED

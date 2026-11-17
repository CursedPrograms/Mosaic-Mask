@echo off
setlocal
cd /d "%~dp0"
rem Installs Mosaic Mask for this user: %LOCALAPPDATA%\Programs\Mosaic Mask, with Start menu and Desktop shortcuts.
rem Run build.bat first (or unzip a release and run this beside it).
set "SRC=dist\MosaicMask"
if exist "MosaicMask.exe" set "SRC=."
if not exist "%SRC%\MosaicMask.exe" ( echo Build it first: build.bat & exit /b 1 )
tasklist /fi "imagename eq MosaicMask.exe" | find /i "MosaicMask.exe" >nul
if not errorlevel 1 ( echo Mosaic Mask is open: close it and run this again. & exit /b 1 )
set "INST=%LOCALAPPDATA%\Programs\Mosaic Mask"
if not exist "%INST%" mkdir "%INST%"
xcopy /e /i /q /y "%SRC%\*" "%INST%\" >nul || ( echo Couldn't copy to %INST% & exit /b 1 )
powershell -NoProfile -Command "$s=New-Object -ComObject WScript.Shell; foreach($d in @([Environment]::GetFolderPath('Programs'),[Environment]::GetFolderPath('Desktop'))){ $l=$s.CreateShortcut((Join-Path $d 'Mosaic Mask.lnk')); $l.TargetPath='%INST%\MosaicMask.exe'; $l.WorkingDirectory='%INST%'; $l.IconLocation='%INST%\MosaicMask.exe,0'; $l.Description='Mosaic Mask: a disguise for video calls'; $l.Save() }"
echo Installed: %INST%
if /i not "%~1"=="--quiet" echo Start it from the Start menu or the Desktop.
exit /b 0

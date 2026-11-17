@echo off
setlocal
cd /d "%~dp0"
rem Builds Mosaic Mask (the window) and mosaic-cli (Release) and dist\MosaicMask-win64.zip.
rem Needs Visual Studio 2022 (C++), CMake and OpenCV 4.10 for Windows (OpenCV_DIR, or D:\cc\_src\opencv-dl\opencv\build);
rem SDL2, Dear ImGui, ONNX Runtime and miniaudio come by themselves on the first build.
if not defined BUILD_DIR set "BUILD_DIR=build"
cmake -B %BUILD_DIR% -G "Visual Studio 17 2022" -A x64 -DMOSAIC_GUI=ON || goto :error
cmake --build %BUILD_DIR% --config Release || goto :error
if exist dist\MosaicMask rmdir /s /q dist\MosaicMask
mkdir dist\MosaicMask
for %%f in (MosaicMask.exe mosaic-cli.exe SDL2.dll onnxruntime.dll opencv_world4100.dll opencv_videoio_ffmpeg4100_64.dll) do copy /y %BUILD_DIR%\Release\%%f dist\MosaicMask\ >nul || goto :error
for %%d in (assets lang models) do xcopy /e /i /q /y %BUILD_DIR%\Release\%%d dist\MosaicMask\%%d >nul
if exist dist\MosaicMask\lang\make_packs.py del dist\MosaicMask\lang\make_packs.py
rem the sample strangers (AI-generated people), copied into Documents\Mosaic Mask\faces on the first start
xcopy /i /q /y ..\faces\*.* dist\MosaicMask\faces >nul
rem Microsoft's Visual C++ runtime
set "CRT="
for /f "usebackq delims=" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -find VC\Redist\MSVC\*\x64\Microsoft.VC143.CRT`) do set "CRT=%%i"
if not defined CRT echo Visual C++ runtime folder not found, is Visual Studio 2022 installed?& goto :error
xcopy /q /y "%CRT%\*.dll" dist\MosaicMask\ >nul || goto :error
for %%f in (..\README.md ..\LICENSE install.bat install.ps1) do if exist %%f copy /y %%f dist\MosaicMask\ >nul
if exist dist\MosaicMask-win64.zip del dist\MosaicMask-win64.zip
"%SystemRoot%\System32\tar.exe" -a -c -f dist\MosaicMask-win64.zip -C dist MosaicMask || goto :error
echo Built: dist\MosaicMask-win64.zip
if exist "%LOCALAPPDATA%\Programs\Mosaic Mask\MosaicMask.exe" call "%~dp0install.bat" --quiet
exit /b 0
:error
echo Build failed, see above.
exit /b 1

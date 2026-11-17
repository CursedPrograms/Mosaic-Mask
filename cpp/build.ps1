# Builds Mosaic Mask and mosaic-cli (Release) and dist\MosaicMask-win64.zip, through build.bat (Visual Studio 2022 + CMake).
$ErrorActionPreference = 'Stop'
& cmd /c "`"$PSScriptRoot\build.bat`""
exit $LASTEXITCODE

# Installs Mosaic Mask for this user (%LOCALAPPDATA%\Programs\Mosaic Mask, Start menu and Desktop shortcuts named "Atelier - Mosaic Mask"), through install.bat.
$ErrorActionPreference = 'Stop'
& cmd /c "`"$PSScriptRoot\install.bat`" $args"
exit $LASTEXITCODE

# run.ps1 - Mosaic Mask. Sets up .\venv the first time (and again when requirements.txt changes), fetches the
# models once, then starts app.py. Options pass through: .\run.ps1 --camera 1 ...
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$Py = Join-Path $PSScriptRoot 'venv\Scripts\python.exe'
function Fail($msg) { Write-Host "Setup failed: $msg"; Read-Host 'Press Enter'; exit 1 }
if (-not (Test-Path $Py)) {
    Write-Host 'Creating venv ...'
    & py -3.11 -m venv venv
    if ($LASTEXITCODE) { Fail 'Mosaic Mask needs Python 3.11.' }
}
$stamp = 'venv\requirements.installed'
if (-not (Test-Path $stamp) -or (Get-FileHash requirements.txt).Hash -ne (Get-FileHash $stamp).Hash) {
    Write-Host 'Installing requirements ...'
    & $Py -m pip install -r requirements.txt
    if ($LASTEXITCODE) { Fail 'pip could not install the requirements.' }
    Copy-Item requirements.txt $stamp -Force
}
function Fetch($file, $url, $what) {
    if (-not (Test-Path "models/$file")) {
        Write-Host "Downloading $what, once ..."
        New-Item -ItemType Directory -Force models | Out-Null
        & curl.exe -L --fail -o "models/$file.part" $url
        if ($LASTEXITCODE) { Fail "could not download $file." }
        Move-Item "models/$file.part" "models/$file" -Force
    }
}
Fetch 'xseg_1.onnx' 'https://huggingface.co/facefusion/models-3.1.0/resolve/main/xseg_1.onnx' 'the occlusion model, about 70 MB'
& $Py app.py @args
if ($LASTEXITCODE) { Read-Host 'Press Enter' }

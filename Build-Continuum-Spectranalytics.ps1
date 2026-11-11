param(
    [string]$Python = (Join-Path $PSScriptRoot '.venv\Scripts\python.exe')
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw 'Create the Python environment and install .[windows-build] first. See README.md.'
}
$Python = (Resolve-Path -LiteralPath $Python).Path
$buildDir = Join-Path $PSScriptRoot 'build\windows'
$distDir = Join-Path $PSScriptRoot 'dist'
$spec = Join-Path $PSScriptRoot 'packaging\windows\Continuum-Spectranalytics.spec'

& $Python -m PyInstaller --noconfirm --distpath $distDir --workpath $buildDir $spec
if ($LASTEXITCODE -ne 0) {
    throw "Windows build failed with exit code $LASTEXITCODE."
}

Get-Item -LiteralPath (Join-Path $distDir 'Continuum Spectranalytics.exe')

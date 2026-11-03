param(
    [switch]$Console
)

$ErrorActionPreference = 'Stop'
$continuumPython = Join-Path $PSScriptRoot '.venv\Scripts\pythonw.exe'
if ($Console) {
    $continuumPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $continuumPython)) {
    throw 'Set up .venv first using the instructions in README.md.'
}

if ($Console) {
    Push-Location -LiteralPath $PSScriptRoot
    try {
        & $continuumPython -m polyrmc.cli gui
    } finally {
        Pop-Location
    }
} else {
    Start-Process -FilePath $continuumPython -ArgumentList '-m', 'polyrmc.cli', 'gui' `
        -WorkingDirectory $PSScriptRoot -WindowStyle Hidden
}

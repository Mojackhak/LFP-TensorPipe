Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$rootDir = Resolve-Path (Join-Path $PSScriptRoot "..\\..")
$condaExe = if ($env:CONDA_EXE) { $env:CONDA_EXE } else { "conda" }

Push-Location $rootDir
try {
    $condaHook = & $condaExe "shell.powershell" "hook"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to initialize the Conda PowerShell hook."
    }
    $condaHook | Out-String | Invoke-Expression
    conda activate lfptp

    python tools/release/build_pyinstaller.py --target-platform windows @args
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}

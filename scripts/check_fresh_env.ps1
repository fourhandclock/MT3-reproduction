param(
    [string]$Python = '',
    [switch]$AllowDownloads
)

$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
if (-not $Python) { $Python = Join-Path $taskRoot '.venv/Scripts/python.exe' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python not found: $Python" }
$taskUv = (Get-Command uv -ErrorAction Stop).Source
$taskRun = Join-Path $taskRoot ('tmp/fresh-transcription/run-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $taskRun -Force | Out-Null
$taskEnvironment = Join-Path $taskRun 'env'
$taskFreshPython = Join-Path $taskEnvironment 'Scripts/python.exe'
$taskSavedEnv = @{}
foreach ($taskKey in @('UV_CACHE_DIR', 'PYTHONPATH', 'PYTHONHOME', 'PYTHONNOUSERSITE', 'TEMP', 'TMP')) {
    $taskSavedEnv[$taskKey] = [Environment]::GetEnvironmentVariable($taskKey, 'Process')
}
$taskOriginalLocation = Get-Location
try {
    Set-Location -LiteralPath $taskRoot
    $env:UV_CACHE_DIR = Join-Path $taskRoot '.cache/uv'
    $env:PYTHONNOUSERSITE = '1'
    $taskTemp = Join-Path $taskRun 'temp'
    New-Item -ItemType Directory -Path $taskTemp | Out-Null
    $env:TEMP = $taskTemp
    $env:TMP = $taskTemp
    Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
    $taskOffline = @('--offline')
    if ($AllowDownloads) { $taskOffline = @() }

    & $taskUv venv @taskOffline --python $Python $taskEnvironment 2>&1 |
        Tee-Object -FilePath (Join-Path $taskRun 'create-env.log')
    if ($LASTEXITCODE -ne 0) { throw 'Could not create isolated environment.' }

    # Reuse the recorded cached CUDA-enabled wheel, but run inference on CPU.
    # No torchvision, torchaudio, MT3 adapter, separator or training packages are needed.
    & $taskUv pip install @taskOffline --python $taskFreshPython 'torch==2.7.1+cu126' `
        --index-url https://download.pytorch.org/whl/cu126 2>&1 |
        Tee-Object -FilePath (Join-Path $taskRun 'install-torch.log')
    if ($LASTEXITCODE -ne 0) { throw 'PyTorch install failed. Offline is the default; see the install log.' }
    & $taskUv pip install @taskOffline --python $taskFreshPython -r requirements-inference.txt 2>&1 |
        Tee-Object -FilePath (Join-Path $taskRun 'install-inference.log')
    if ($LASTEXITCODE -ne 0) { throw 'Inference dependencies could not be installed; see the install log.' }
    & $taskUv pip check --python $taskFreshPython 2>&1 |
        Tee-Object -FilePath (Join-Path $taskRun 'dependencies.log')
    if ($LASTEXITCODE -ne 0) { throw 'Fresh environment has incompatible dependencies.' }
    & $taskUv pip freeze --python $taskFreshPython |
        Set-Content -LiteralPath (Join-Path $taskRun 'requirements-resolved.txt') -Encoding UTF8
    if ($LASTEXITCODE -ne 0) { throw 'Could not record installed versions.' }
    & $taskFreshPython scripts/smoke_transcription.py --output-dir $taskRun
    if ($LASTEXITCODE -ne 0) { throw 'Fresh environment transcription smoke failed; see smoke.json and transcription.log.' }
    Write-Output "Fresh environment evidence: $taskRun"
}
finally {
    Set-Location -LiteralPath $taskOriginalLocation
    foreach ($taskKey in $taskSavedEnv.Keys) {
        [Environment]::SetEnvironmentVariable($taskKey, $taskSavedEnv[$taskKey], 'Process')
    }
}

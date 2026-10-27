param([switch]$CpuOnly)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $taskRoot
$env:UV_CACHE_DIR = Join-Path $taskRoot '.cache\uv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $taskRoot '.cache\python'
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    uv venv --python 3.10 .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python environment setup failed.' }
}
$taskTorchIndex = if ($CpuOnly) { 'https://download.pytorch.org/whl/cpu' } else { 'https://download.pytorch.org/whl/cu126' }
uv pip install --python .venv\Scripts\python.exe torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url $taskTorchIndex
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed.' }
uv pip install --python .venv\Scripts\python.exe -r requirements-baseline.txt
if ($LASTEXITCODE -ne 0) { throw 'Audio dependencies installation failed.' }
if (-not (Test-Path -LiteralPath 'third_party\mt3-infer')) {
    git clone https://github.com/openmirlab/mt3-infer.git third_party/mt3-infer
    if ($LASTEXITCODE -ne 0) { throw 'MT3 port download failed.' }
}
git -C third_party/mt3-infer checkout d8cbd287470291dee335f84315c5a9b9ff6e6c8a
if ($LASTEXITCODE -ne 0) { throw 'Could not select the recorded MT3 inference revision.' }
uv pip install --python .venv\Scripts\python.exe --no-deps basic-pitch==0.4.0 -e third_party/mt3-infer
if ($LASTEXITCODE -ne 0) { throw 'Model adapter installation failed.' }
uv pip freeze --python .venv\Scripts\python.exe | Set-Content -LiteralPath requirements-resolved.txt -Encoding utf8
Write-Output 'Setup complete. Run .venv\Scripts\python.exe -m baseline doctor'

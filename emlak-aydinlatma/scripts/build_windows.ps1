#Requires -Version 5.1
<#
.SYNOPSIS
  EmlakAydinlatma Windows one-folder exe derlemesi (PyInstaller).

.DESCRIPTION
  Bu script Windows üzerinde çalışmalıdır (Linux VM'de Windows .exe üretilmez).
  Çıktı: emlak-aydinlatma\dist\EmlakAydinlatma\EmlakAydinlatma.exe

.EXAMPLE
  cd emlak-aydinlatma
  .\scripts\build_windows.ps1
#>

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

Write-Host "==> Klasör: $Root" -ForegroundColor Cyan

# Python kontrol
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) {
    Write-Error "Python bulunamadı. https://www.python.org/downloads/ adresinden kurun (Add to PATH işaretli)."
}

Write-Host "==> Python: $(python --version)" -ForegroundColor Cyan

# venv
$venvPath = Join-Path $Root ".venv"
if (-not (Test-Path (Join-Path $venvPath "Scripts\python.exe"))) {
    Write-Host "==> Sanal ortam oluşturuluyor (.venv)..." -ForegroundColor Cyan
    python -m venv .venv
}

$venvPy = Join-Path $venvPath "Scripts\python.exe"
$venvPip = Join-Path $venvPath "Scripts\pip.exe"

Write-Host "==> Bağımlılıklar kuruluyor..." -ForegroundColor Cyan
& $venvPip install --upgrade pip
& $venvPip install -r (Join-Path $Root "requirements.txt")
& $venvPip install -r (Join-Path $Root "requirements-build.txt")

# Tkinter smoke
Write-Host "==> Tkinter kontrolü..." -ForegroundColor Cyan
& $venvPy -c "import tkinter; print('tkinter OK')"

# Import freeze kontrolü
Write-Host "==> Paket import kontrolü..." -ForegroundColor Cyan
$env:PYTHONPATH = Join-Path $Root "src"
& $venvPy (Join-Path $Root "scripts\verify_freeze_imports.py")
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue

# Temiz derleme
Write-Host "==> PyInstaller (one-folder)..." -ForegroundColor Cyan
$pyi = Join-Path $venvPath "Scripts\pyinstaller.exe"
& $pyi --noconfirm --clean (Join-Path $Root "emlak_aydinlatma.spec")

$exe = Join-Path $Root "dist\EmlakAydinlatma\EmlakAydinlatma.exe"
if (-not (Test-Path $exe)) {
    Write-Error "Derleme başarısız: $exe bulunamadı."
}

Write-Host ""
Write-Host "Tamam. Çalıştırılabilir dosya:" -ForegroundColor Green
Write-Host "  $exe"
Write-Host ""
Write-Host "Kullanım:" -ForegroundColor Green
Write-Host "  1) dist\EmlakAydinlatma klasörünün tamamını kopyalayın (sadece .exe yetmez)."
Write-Host "  2) EmlakAydinlatma.exe dosyasına çift tıklayın."
Write-Host "  3) İlk açılışta Windows Defender uyarısı çıkabilir → Daha fazla bilgi → Yine de çalıştır."
Write-Host ""

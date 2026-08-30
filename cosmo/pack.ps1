# Buduje BoneCosmo.exe i przenośny folder dist\
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
cargo build --release
$out = Join-Path $PSScriptRoot "dist\BoneCosmo"
New-Item -ItemType Directory -Force -Path $out | Out-Null
Copy-Item "target\release\BoneCosmo.exe" "$out\BoneCosmo.exe" -Force
Copy-Item "README.md" "$out\README.md" -Force
$wix = "C:\Program Files\WiX Toolset v6.0\bin\wix.exe"
if (Test-Path $wix) {
  & $wix build wix\Package.wxs -o dist\BoneCosmo.msi -arch x64 -b app=dist\BoneCosmo
  Write-Host "MSI: $PSScriptRoot\dist\BoneCosmo.msi"
}
Write-Host "Portable: $out\BoneCosmo.exe"

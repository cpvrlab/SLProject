$GitPath = Get-Command Git | Select Path | Split-Path -Parent | Split-Path -Parent
$Shell = $GitPath + "\bin\sh.exe"
& $Shell .\build-mediapipe-x86_64-windows.sh @args
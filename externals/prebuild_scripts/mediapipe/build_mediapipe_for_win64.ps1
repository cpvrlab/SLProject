$GitPath = Get-Command Git | Select Path | Split-Path -Parent | Split-Path -Parent
$Shell = $GitPath + "\bin\sh.exe"
& $Shell .\build_mediapipe_for_win64.sh @args
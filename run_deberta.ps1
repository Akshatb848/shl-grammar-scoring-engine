# Fine-tune DeBERTa over 5 folds, restarting at the next unfinished fold if the process dies.
# Usage: powershell -File run_deberta.ps1 -Model medium.en
param([string]$Model = "medium.en", [string]$Extra = "")
Set-Location $PSScriptRoot
$log = "run_deberta_$Model$Extra.log"
"=== start $(Get-Date)" | Out-File -Append -Encoding utf8 $log
for ($attempt = 1; $attempt -le 10; $attempt++) {
    "--- attempt $attempt $(Get-Date)" | Out-File -Append -Encoding utf8 $log
    # cmd redirection keeps the log in UTF-8 (PowerShell's *>> would write UTF-16).
    cmd /c ".venv\Scripts\python.exe -u finetune_deberta.py $Model $Extra >> $log 2>&1"
    if ($LASTEXITCODE -eq 0) { break }
    Start-Sleep -Seconds 10
}
"=== exit $LASTEXITCODE $(Get-Date)" | Out-File -Append -Encoding utf8 $log

# Full pipeline for one Whisper model, each stage restarting automatically if its process dies:
#   1. transcription   2. DeBERTa, 3 seeds x 5 folds   3. notebook run into outputs_<model>/ and <notebook>_<model>.ipynb
# Usage: powershell -File run_all.ps1 -Model large-v3
param([string]$Model = "medium.en", [int]$Seeds = 3)
Set-Location $PSScriptRoot
$log = "run_all_$Model.log"
function Log($msg) { "$msg $(Get-Date)" | Out-File -Append -Encoding utf8 $log }

Log "=== start"
for ($attempt = 1; $attempt -le 20; $attempt++) {
    Log "--- transcribe attempt $attempt"
    cmd /c ".venv\Scripts\python.exe -u transcribe.py $Model >> $log 2>&1"
    if ($LASTEXITCODE -eq 0) { break }
    Start-Sleep -Seconds 10
}

for ($s = 0; $s -lt $Seeds; $s++) {
    Log "--- deberta seed $s"
    & .\run_deberta.ps1 -Model $Model -Extra "--seed=$s"
}

Log "--- notebook"
$env:WHISPER_MODEL = $Model; $env:DROP_ZERO = "0"; $env:DEBERTA_SEEDS = "$Seeds"; $env:OUTPUT_DIR = "outputs_$Model"
Copy-Item shl_grammar_scoring.ipynb "shl_grammar_scoring_$Model.ipynb" -Force
cmd /c ".venv\Scripts\jupyter.exe nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 shl_grammar_scoring_$Model.ipynb >> $log 2>&1"
Log "=== notebook exit $LASTEXITCODE"

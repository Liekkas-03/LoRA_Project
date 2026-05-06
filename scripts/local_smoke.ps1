$ErrorActionPreference = "Stop"

Write-Host "Running lightweight local smoke checks..."

$Python = "python"
$AnacondaPython = "D:\Anaconda3\python.exe"
if (Test-Path $AnacondaPython) {
    $Python = $AnacondaPython
}

& $Python -m src.eval.evaluate_accuracy --input data\samples\math_preds.jsonl
& $Python -m src.eval.error_classifier --input data\samples\math_preds.jsonl --output outputs\reports\sample_errors.jsonl
& $Python -m src.data.difficulty_scorer --input data\samples\math_preds.jsonl --output outputs\reports\sample_scored.jsonl

Write-Host "Smoke checks finished."

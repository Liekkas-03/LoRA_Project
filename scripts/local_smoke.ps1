$ErrorActionPreference = "Stop"

# 本脚本只做本地轻量检查：MATH 样例转换、划分和评测，不安装任何依赖。
Write-Host "Running MATH-only lightweight smoke checks..."

$Python = "python"
$AnacondaPython = "D:\Anaconda3\python.exe"
if (Test-Path $AnacondaPython) {
    $Python = $AnacondaPython
}

& $Python -m src.data.prepare_math --input-dir data\samples\math_raw --output outputs\reports\math_prepared_sample.jsonl --split train
& $Python -m src.data.split_data --input outputs\reports\math_prepared_sample.jsonl --train-output outputs\reports\math_train_sample.jsonl --dev-output outputs\reports\math_dev_sample.jsonl --dev-ratio 0.5 --seed 42 --stratify-field difficulty_group
& $Python -m src.eval.evaluate_accuracy --input data\samples\math_preds.jsonl
& $Python -m src.data.difficulty_scorer --input data\samples\math_preds.jsonl --output outputs\reports\math_scored_sample.jsonl

Write-Host "MATH-only smoke checks finished."

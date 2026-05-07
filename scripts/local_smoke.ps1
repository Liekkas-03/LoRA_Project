$ErrorActionPreference = "Stop"

# 本脚本负责本地轻量冒烟测试，只运行标准库数据处理和评测脚本，不安装任何依赖。
Write-Host "Running lightweight local smoke checks..."

$Python = "python"
$AnacondaPython = "D:\Anaconda3\python.exe"
if (Test-Path $AnacondaPython) {
    $Python = $AnacondaPython
}

& $Python -m src.data.prepare_gsm8k --input data\samples\gsm8k_raw.jsonl --output outputs\reports\gsm8k_prepared_sample.jsonl --split train
& $Python -m src.data.prepare_math --input-dir data\samples\math_raw --output outputs\reports\math_prepared_sample.jsonl --split train
& $Python -m src.data.split_data --input outputs\reports\gsm8k_prepared_sample.jsonl --train-output outputs\reports\gsm8k_train_sample.jsonl --dev-output outputs\reports\gsm8k_dev_sample.jsonl --dev-ratio 0.5 --seed 42
& $Python -m src.data.merge_data --inputs outputs\reports\gsm8k_train_sample.jsonl outputs\reports\math_prepared_sample.jsonl --output outputs\reports\train_sample.jsonl
& $Python -m src.eval.evaluate_accuracy --input data\samples\math_preds.jsonl
& $Python -m src.eval.error_classifier --input data\samples\math_preds.jsonl --output outputs\reports\sample_errors.jsonl
& $Python -m src.data.difficulty_scorer --input data\samples\math_preds.jsonl --output outputs\reports\sample_scored.jsonl

Write-Host "Smoke checks finished."

#!/usr/bin/env bash
# Quick-start: install deps → train → evaluate → plot
set -e
cd "$(dirname "$0")"

echo "=== Installing dependencies ==="
pip install -r requirements.txt -q

echo ""
echo "=== Training (synthetic data, 50 epochs max) ==="
python -m src.train --config configs/default.yaml --output_dir outputs

echo ""
echo "=== Evaluation ==="
python -m src.evaluate --checkpoint outputs/best_model.pt

echo ""
echo "=== Generating figures ==="
python -m src.visualize --checkpoint outputs/best_model.pt --outdir outputs/figures

echo ""
echo "=== Inference demo ==="
python -m src.inference --checkpoint outputs/best_model.pt --n_samples 3

echo ""
echo "Done!  Check outputs/ for model, metrics and figures."

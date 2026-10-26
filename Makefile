.PHONY: install serve train samples test evaluate
install:
	python -m pip install -e '.[dev]'
serve:
	python -m realityforge.cli serve
train:
	python scripts/train.py --count 6000 --epochs 100 --seed 47
samples:
	python scripts/generate_samples.py
test:
	python -m pytest -q
evaluate:
	python scripts/evaluate_synthetic.py --seed 2026 --cases 24
	python scripts/evaluate_real.py

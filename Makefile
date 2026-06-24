.PHONY: install run run-edge debug clean lint lint-strict

install:
	uv venv
	uv sync

run:
	HF_TOKEN="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" uv run python -m src

run-edge:
	uv run python -m src --input data/edge/edge_cases.json

debug:
	.venv/bin/python3 -m pdb fly_in.py

clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .DS_Store

lint:
	.venv/bin/flake8 src
	.venv/bin/mypy src --follow-imports=silent --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	.venv/bin/flake8 src
	.venv/bin/mypy src --follow-imports=silent --strict
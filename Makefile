.PHONY: install fetch-files-home fetch-files-42 run run-edge debug clean lint lint-strict

install:
	uv venv
	uv sync

fetch-files-home:
	rm -f en.subject.pdf
	rm -f data.zip
	rm -f llm_sdk.zip
	curl -O https://cdn.intra.42.fr/pdf/pdf/206639/en.subject.pdf
	curl -O https://cdn.intra.42.fr/document/document/49768/data.zip
	curl -O https://cdn.intra.42.fr/document/document/49769/llm_sdk.zip
	unzip -u data.zip
	unzip -u llm_sdk.zip
	rm -f data.zip
	rm -f llm_sdk.zip

fetch-files-42:
	rm -f en.subject.pdf
	rm -f data.zip
	rm -f llm_sdk.zip
	wget https://cdn.intra.42.fr/pdf/pdf/206639/en.subject.pdf
	wget https://cdn.intra.42.fr/document/document/49768/data.zip
	wget https://cdn.intra.42.fr/document/document/49769/llm_sdk.zip
	unzip -u data.zip
	unzip -u llm_sdk.zip
	rm -f data.zip
	rm -f llm_sdk.zip

run:
	uv run python -m src

run-edge:
	uv run python -m src --input data/edge/edge_cases.json

debug:
	uv run python -m pdb -m src

clean:
	rm -rf __pycache__ src/__pycache__ moulinette/__pycache__
	rm -rf .pytest_cache .mypy_cache
	rm -rf data/output

lint:
	.venv/bin/flake8 . --exclude=.venv,moulinette,llm_sdk
	.venv/bin/mypy . --exclude .venv,moulinette,llm_sdk --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	.venv/bin/flake8 . --exclude=.venv,moulinette,llm_sdk
	.venv/bin/mypy . --exclude .venv,moulinette,llm_sdk --strict
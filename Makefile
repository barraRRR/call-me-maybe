.PHONY: install run run-edge debug clean lint

install:
		uv venv
		uv sync
		rm -f en.subject.pdf
		uv run python -m wget https://cdn.intra.42.fr/pdf/pdf/206639/en.subject.pdf
		uv run python -m wget https://cdn.intra.42.fr/document/document/49768/data.zip
		uv run python -m wget https://cdn.intra.42.fr/document/document/49769/llm_sdk.zip
		unzip -u data.zip
		unzip -u llm_sdk.zip
		rm -f data.zip
		rm -f llm_sdk.zip
		uv pip install -e ./llm_sdk

run:
		uv run python -m src

run-edge:
		uv run python -m src --input data/edge/edge_cases.json

debug:
		.venv/bin/python3 -m pdb fly_in.py

clean:
		rm -rf __pycache__
		rm -rf .pytest_cache
		rm -rf .mypy_cache
		rm output_file.txt

lint:
		.venv/bin/flake8 . --exclude=.venv,moulinette,llm_sdk && .venv/bin/mypy . --exclude .venv,moulinette,llm_sdk --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
		.venv/bin/flake8 . --exclude=.venv,moulinette,llm_sdk && .venv/bin/mypy . --exclude .venv,moulinette,llm_sdk --strict



		
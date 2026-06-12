.PHONY: install run debug clean lint

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



		
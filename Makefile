.PHONY: install, run

install:
		python3 -m venv venv
		./venv/bin/python3 -m pip install --upgrade pip
		./venv/bin/python3 -m pip install -r requirements.txt
		./venv/bin/python3 -m wget https://cdn.intra.42.fr/document/document/49768/data.zip
		./venv/bin/python3 -m wget https://cdn.intra.42.fr/document/document/49769/llm_sdk.zip
		unzip -u data.zip
		unzip -u llm_sdk.zip
		rm data.zip
		rm llm_sdk.zip



		
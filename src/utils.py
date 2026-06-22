from typing import List, Dict, Any
import os
import json


TOKENIZER_PATH = "src/tokenizer.json"
BASE_PROMPT_PATH = "src/base_prompt.txt"
FUNC_DEF_PATH = "data/input/functions_definition.json"
FUNC_CALL_TESTS_PATH = "data/input/function_calling_tests.json"
OUTPUT_PATH = "data/output/function_calls.json"
EOS_TOKEN_ID = 151645


def compose_output_file(
        model_response_list: List[Dict[str, Any]],
        output_path: str = OUTPUT_PATH) -> None:
    """
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            model_response_list,
            fp=f,
            indent=2,
            ensure_ascii=False
            )


def debug_output_token_list(debug_token_list: List[str]) -> None:
    """
    """
    debug_toke_list_path: str = "data/debug/debug_token_list.json"
    os.makedirs(os.path.dirname(debug_toke_list_path), exist_ok=True)
    with open(debug_toke_list_path, "w", encoding="utf-8") as f:
        json.dump(
            debug_token_list,
            fp=f,
            indent=2,
            ensure_ascii=False
            )
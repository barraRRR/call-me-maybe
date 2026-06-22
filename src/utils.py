from typing import List, Dict, Any
import os
import json
import re


TOKENIZER_PATH = "src/tokenizer.json"
BASE_PROMPT_PATH = "src/base_prompt.txt"
FUNC_DEF_PATH = "data/input/functions_definition.json"
FUNC_CALL_TESTS_PATH = "data/input/function_calling_tests.json"
OUTPUT_PATH = "data/output/function_calls.json"
ERROR_MSG_PATH = "src/error_handling.json"
EOS_TOKEN_ID = 151645


number_regex = re.compile(
    r'^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$'
)


def is_quote_escaped(s: str, i: int) -> bool:
    count = 0
    j = i - 1
    while j >= 0 and s[j] == '\\':
        count += 1
        j -= 1
    return count % 2 != 0


def find_first_unescaped_quote(s: str) -> int:
    for i, c in enumerate(s):
        if c == '"' and not is_quote_escaped(s, i):
            return i
    return -1


def is_complete_number(s: str) -> bool:
    try:
        float(s)
        if s in ('', '+', '-', '.', '+.', '-.', 'e', 'E', '+e', '-e'):
            return False
        if not s[-1].isdigit() and s[-1] not in ('.', 'e', 'E'):
            return False
        return True
    except ValueError:
        return False


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

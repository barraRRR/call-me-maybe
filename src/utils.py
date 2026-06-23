from typing import List, Dict, Any, Optional
import sys
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


def clear() -> None:
    """Clears the console or terminal screen cleanly."""
    os.system("cls" if os.name == "nt" else "clear")


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


def title(center: int = 100) -> str:
    """Generates the ASCII title graphic for game screens.

    Returns:
        str: Centered multiline ASCII string.
    """
    ascii_art = r"""           ____                              __      
 _______ _/ / / __ _  ___   __ _  ___ ___ __/ /  ___ 
/ __/ _ `/ / / /  ' \/ -_) /  ' \/ _ `/ // / _ \/ -_)
\__/\_,_/_/_/ /_/_/_/\__/ /_/_/_/\_,_/\_, /_.__/\__/ 
                                     /___/           """
    return ascii_art.center(center)


def wait_for_enter(message: Optional[str]) -> None:
    """Halts execution until the user presses the 'Enter' key.

    Args:
        message (str, optional): Custom override string to display.
    """
    if message is None:
        message = "Press [ENTER] to continue..."
    input(message)


def welcome() -> None:
    """Displays the interactive title graphic for program launch."""
    clear()
    print()
    print(title(), end="\n" * 3)
    wait_for_enter(None)


def goodbye() -> str:
    """Exits the application gracefully displaying parting graphics."""
    goodby_msg = "Thanks for evaluating call-me-maybe!"
    clear()
    print("\n\n")
    print(title(), end="\n" * 3)
    print(goodby_msg, "\n")
    sys.exit(0)


def print_error(
        error_msg: str = "An unexpected error ocurred...",
        critical: bool = False, 
        error_title: str = "ERROR") -> None:
    """
    """
    error_title = error_title if not critical else "CRITICAL ERROR"
    error_subtitle = "A critical error ocurred" if critical else "An unexpected error ocurred" 
    comp_msg = (
        "\n"
        f"  [{error_title}]: {error_subtitle}\n"
        f"      └ {error_msg}\n"
    )
    print(comp_msg)
    
    if critical:
        wait_for_enter()
        goodbye()

from typing import List, Dict, Any
import sys
import os
import json
import re


TOKENIZER_PATH = "src/tokenizer.json"
BASE_PROMPT_PATH = "src/BASE_PROMPT.txt"
FUNC_DEF_PATH = "data/input/functions_definition.json"
FUNC_CALL_TESTS_PATH = "data/input/function_calling_tests.json"
OUTPUT_PATH = "data/output/function_calls.json"
ERROR_MSG_PATH = "src/error_handling.json"
EOS_TOKEN_ID = 151645
UX_WIDTH = 100


number_regex = re.compile(
    r'^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$'
)


def clear() -> None:
    """Clears the console or terminal screen cleanly."""
    os.system("cls" if os.name == "nt" else "clear")


def is_quote_escaped(s: str, i: int) -> bool:
    """Checks if the double quote character at index i is escaped.

    Args:
        s (str): The string to check.
        i (int): The index of the quote character.

    Returns:
        bool: True if the quote is escaped by an odd number of backslashes,
            False otherwise.
    """
    count = 0
    j = i - 1
    while j >= 0 and s[j] == '\\':
        count += 1
        j -= 1
    return count % 2 != 0


def find_first_unescaped_quote(s: str) -> int:
    """Finds the index of the first unescaped double quote in a string.

    Args:
        s (str): The string to search.

    Returns:
        int: The 0-based index of the first unescaped quote,
            or -1 if not found.
    """
    for i, c in enumerate(s):
        if c == '"' and not is_quote_escaped(s, i):
            return i
    return -1


def is_complete_number(s: str) -> bool:
    """Validates if a string represents a fully formed numeric value.

    Args:
        s (str): The string to evaluate.

    Returns:
        bool: True if the string is a complete numeric representation,
            False otherwise.
    """
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
    """Writes the model responses to a structured JSON output file.

    Args:
        model_response_list (List[Dict[str, Any]]): The list of model
            predictions to save.
        output_path (str): The destination file path. Defaults to OUTPUT_PATH.
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
    """Saves the tracked generation tokens to a debug file for logging.

    Args:
        debug_token_list (List[str]): The list of generated token string
            representations.
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


def title(center: int = UX_WIDTH) -> str:
    """Generates the ASCII title graphic for game screens.

    Returns:
        str: Centered multiline ASCII string.
    """
    ascii_art = r"""            ____                              __
  _______ _/ / / __ _  ___   __ _  ___ ___ __/ /  ___
 / __/ _ `/ / / /  ' \/ -_) /  ' \/ _ `/ // / _ \/ -_)
 \__/\_,_/_/_/ /_/_/_/\__/ /_/_/_/\_,_/\_, /_.__/\__/
                                      /___/           """
    return ascii_art.center(center)


def wait_for_enter(
        message: str = "Press [ENTER] to continue...") -> None:
    """Halts execution until the user presses the 'Enter' key.

    Args:
        message (str, optional): Custom override string to display.
    """
    input(message)
    print("\033[F\033[2K", end="")
    print("\r", end="")


def welcome() -> None:
    """Displays the interactive title graphic for program launch."""
    clear()
    print()
    print(title(), end="\n" * 3)
    wait_for_enter()
    clear()


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
    """Formats and prints an error message to the console.

    If critical is True, this function halts execution until the user
    presses Enter and then terminates the program.

    Args:
        error_msg (str): The details of the error message.
        critical (bool): If True, halts execution and exits the program.
            Defaults to False.
        error_title (str): The visual title of the error banner.
            Defaults to "ERROR".
    """
    error_title = error_title if not critical else "CRITICAL ERROR"
    error_subtitle = (
        "A critical error ocurred" if critical
        else "An unexpected error ocurred"
    )
    comp_msg = (
        "\n"
        f"  [{error_title}]: {error_subtitle}\n"
        f"      └ {error_msg}\n"
    )
    print(comp_msg)

    if critical:
        wait_for_enter()
        goodbye()


def section_header(header: str, width: int = UX_WIDTH) -> None:
    """Prints a structured section header container.

    Args:
        header (str): Title name for the header container.
        width (int): Target print visual character width constraint.
            Defaults to UX_WIDTH.
    """
    header_str: str = (
        "\n+" + "-" * (width - 2) + "+\n" +
        "|" + f"{header.center(width - 2)}" + "|\n" +
        "+" + "-" * (width - 2) + "+\n"
    )
    print(header_str)

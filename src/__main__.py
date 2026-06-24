from llm_sdk.llm_sdk import Small_LLM_Model
from typing import Optional, Dict, List, Tuple, Any, Protocol, Set, cast
from pydantic import BaseModel, create_model, ValidationError
from pydantic import Field, PrivateAttr, ConfigDict
from src.utils import TOKENIZER_PATH, BASE_PROMPT_PATH, FUNC_DEF_PATH
from src.utils import OUTPUT_PATH, FUNC_CALL_TESTS_PATH, EOS_TOKEN_ID
from src.utils import compose_output_file, UX_WIDTH
from src.utils import is_quote_escaped, is_complete_number, ERROR_MSG_PATH
from src.utils import find_first_unescaped_quote, number_regex
from src.utils import welcome, goodbye, print_error
from src.utils import section_header
from src import __description__
from enum import Enum
from time import sleep
import numpy as np
import json
import argparse
import os


def load_error_messages(path: str = ERROR_MSG_PATH) -> Dict[str, str]:
    """Loads error templates from JSON or returns a fallback mapping.

    Args:
        path (str): The error messages file path. Defaults to ERROR_MSG_PATH.

    Returns:
        Dict[str, str]: The error messages dictionary.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return cast(Dict[str, str], json.load(f))
    except Exception:
        print_error(
            error_msg="error_handling.json missing or invalid.", critical=True
        )
        return {}


ERRORS = load_error_messages()


class ParameterInfo(BaseModel):
    """Schema description for a single parameter.

    Attributes:
        type (str): The expected data type of the parameter.
    """
    type: str


class FunctionCallResult(BaseModel):
    """Data model representing a structured function call prediction.

    Attributes:
        prompt (str): The original natural-language prompt.
        name (str): The name of the function to invoke.
        parameters (BaseModel): The validated parameter model.
    """
    prompt: str
    name: str
    parameters: BaseModel


class FunctionSchema(BaseModel):
    """Schema descriptor containing complete metadata for a function.

    Attributes:
        name (str): The unique name of the function.
        description (str): Explanatory description of the function's behavior.
        parameters (Dict[str, ParameterInfo]): Parameter descriptors.
        returns (Dict[str, str]): Return type descriptor.
    """
    name: str
    description: str
    parameters: Dict[str, ParameterInfo]
    returns: Dict[str, str]


class DynamicFunctionDefinitions(BaseModel):
    """Handles parsing of functions definitions JSON and compiles validators.

    Attributes:
        func_def_path (str): File path to the function definitions JSON.
        func_def_dict (Dict[str, FunctionSchema]): Map of loaded schemas.
        validators (Dict[str, Any]): Dynamic Pydantic validator models.
    """
    func_def_path: str
    func_def_dict: Dict[str, FunctionSchema] = Field(default_factory=dict)
    validators: Dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        """Initializes function schemas and generates validators.

        Runs post-instantiation.
        """
        self._load_functions_definition()
        self._preconfigure_validators()

    def _load_functions_definition(self) -> None:
        """Loads and parses function definitions from the JSON file path.

        Raises:
            FileNotFoundError: If the function definitions file does not exist.
            JSONDecodeError: If the file is not valid JSON.
            ValidationError: If the loaded schema fails validation rules.
        """
        try:
            with open(self.func_def_path, "r") as file:
                raw = json.load(file)
                func_def_list: List[FunctionSchema] = [
                    FunctionSchema(**f) for f in raw
                ]
                self.func_def_dict.update({
                    func.name: func for func in func_def_list
                })

        except FileNotFoundError:
            error_msg = (
                ERRORS["func_def_not_found"].format(path=self.func_def_path)
            )
            print_error(error_msg=error_msg, critical=True)

        except json.JSONDecodeError:
            error_msg = (
                ERRORS["func_def_json_error"].format(path=self.func_def_path)
            )
            print_error(error_msg=error_msg, critical=True)

        except ValidationError as e:
            error_msg = ERRORS["func_def_validation_error"].format(
                path=self.func_def_path, error=e
            )
            print_error(error_msg=error_msg, critical=True)

    def _preconfigure_validators(self) -> None:
        """Compiles dynamic Pydantic validation models."""
        type_map: Dict[str, Any] = {
            "number": float,
            "int": int,
            "integer": int,
            "float": float,
            "string": str,
            "str": str,
            "boolean": bool,
            "bool": bool
        }

        for func_name, func_schema in self.func_def_dict.items():
            param_fields: Dict[str, Any] = {}

            for param_name, param_info in func_schema.parameters.items():
                python_type = type_map.get(param_info.type, Any)
                param_fields[param_name] = (python_type, ...)

            ParamDynamicModel: BaseModel = create_model(
                f"params_{func_name}",
                **param_fields
            )
            self.validators[func_name] = ParamDynamicModel


class UserPrompts(BaseModel):
    """Loads and holds the natural language prompts to test the model against.

    Attributes:
        test_prompts_path (str): File path to the test prompts JSON file.
    """
    test_prompts_path: str
    _prompt_list: List[str] = PrivateAttr(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        """Loads the test prompts post instantiation."""
        self._load_user_prompts()

    def _load_user_prompts(self) -> None:
        """Parses the test prompts JSON file and stores the prompts list.

        Raises:
            FileNotFoundError: If the file does not exist.
            JSONDecodeError: If the file format is invalid.
        """
        try:
            with open(self.test_prompts_path, "r") as file:
                raw: List[Dict[str, str]] = json.load(file)
                self._prompt_list.extend([p["prompt"] for p in raw])

        except FileNotFoundError:
            error_msg = (
                ERRORS["user_prompts_not_found"].format(
                    path=self.test_prompts_path
                )
            )
            print_error(error_msg=error_msg, critical=True)

        except json.JSONDecodeError:
            error_msg = (
                ERRORS["user_prompts_json_error"].format(
                    path=self.test_prompts_path
                )
            )
            print_error(error_msg=error_msg, critical=True)

    def get(self) -> List[str]:
        """Gets the list of loaded test prompts.

        Returns:
            List[str]: The list of test prompts.
        """
        return self._prompt_list


class BasePrompt(BaseModel):
    """Manages formulation and injection of context into base LLM prompt.

    Attributes:
        func_def_dict (Dict[str, FunctionSchema]): Available definitions.
        prompt_path (str): File path to prompt template.
            Defaults to BASE_PROMPT_PATH.
        base_prompt_str (str): The raw base prompt string loaded from file.
        composed_base_prompt (str): Base prompt with schemas injected.
    """
    func_def_dict: Dict[str, FunctionSchema]
    prompt_path: str = BASE_PROMPT_PATH
    base_prompt_str: str = ""
    composed_base_prompt: str = ""

    def model_post_init(self, __context: Any) -> None:
        """Loads and formats the prompt templates post instantiation."""
        self._load_base_prompt()
        self._inject_func_def()

    def _load_base_prompt(self) -> None:
        """Loads the raw base prompt template from file.

        Raises:
            FileNotFoundError: If the template file is missing.
        """
        try:
            with open(self.prompt_path, "r", encoding="utf-8") as f:
                self.base_prompt_str = f.read()

        except FileNotFoundError:
            error_msg = ERRORS["base_prompt_not_found"].format(
                path=self.prompt_path
            )
            print_error(error_msg=error_msg, critical=True)

    def _inject_func_def(self) -> None:
        """Injects JSON representations of available function definitions."""
        funcs_to_inject = [
            f.model_dump() for name, f in self.func_def_dict.items()
        ]
        func_def_text: str = json.dumps(
            funcs_to_inject,
            separators=(',', ':'),
            ensure_ascii=False
        )
        self.base_prompt_str = self.base_prompt_str.format(
            functions_context=func_def_text
        )

    def compose_base_prompt(self, user_promp_str: str) -> str:
        """Composes the final prompt containing context and query.

        Args:
            user_promp_str (str): The user query prompt.

        Returns:
            str: The fully composed model query prompt.
        """
        self.composed_base_prompt = (
            self.base_prompt_str + '"' + user_promp_str +
            '"\n' + "JSON Output:"
        )
        return self.composed_base_prompt

    def get(self) -> str:
        """Gets the composed base prompt.

        Returns:
            str: The composed base prompt string.
        """
        return self.composed_base_prompt


class TokenizerMaskProtocol(Protocol):
    """Protocol defining the interface for generation trackers."""

    def mask(
        self,
        generated_text: str,
        logits: List[float],
        id_to_token: Dict[int, str]
    ) -> List[float]:
        """Masks output logits based on constraints and current context.

        Args:
            generated_text (str): The currently decoded text slice.
            logits (List[float]): Logits list from the model.
            id_to_token (Dict[int, str]): Token vocabulary dictionary.

        Returns:
            List[float]: The modified logits list with invalid tokens masked.
        """
        ...

    def end_condition(self, current_text: str) -> bool:
        """Checks if the end conditions are met to terminate generation.

        Args:
            current_text (str): The currently generated string.

        Returns:
            bool: True if generation should terminate, False otherwise.
        """
        ...


class JSONState(Enum):
    KEY = "KEY"
    KEY_START = "KEY_START"
    KEY_PARTIAL = "KEY_PARTIAL"
    COLON_START = "COLON_START"
    COLON = "COLON"
    VALUE_START = "VALUE_START"
    VALUE = "VALUE"
    VALUE_PARTIAL = "VALUE_PARTIAL"
    COMMA_OR_CLOSE_START = "COMMA_OR_CLOSE_START"
    COMMA_OR_CLOSE = "COMMA_OR_CLOSE"
    CLOSED = "CLOSED"
    ERROR = "ERROR"


class ConstrainedJSONTracker(BaseModel):
    """State-machine based JSON state tracker for constrained decoding.

    Attributes:
        func_def (DynamicFunctionDefinitions): Functions schema catalog.
        user_prompt (str): The natural language query. Defaults to empty.
        model (Any): The model engine. Defaults to None.
        prefix (str): Start template boundary representation.
        prefix_tokens (List[int]): Pre-tokenized prompt suffix prefix list.
        decoded_prefix (str): Decoded representation of prefix sequence.
        param_suffix (str): Serialized start of parameter keys.
        fn_suffix_tokens (Dict[str, List[int]]): Pre-tokenized function suffix.
        id_to_token (Dict[int, str]): Reverse mapped token dictionary.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    func_def: DynamicFunctionDefinitions
    user_prompt: str = ""
    model: Any = None
    prefix: str = ""
    prefix_tokens: List[int] = Field(default_factory=list)
    decoded_prefix: str = ""
    param_suffix: str = '","parameters":{'
    fn_suffix_tokens: Dict[str, List[int]] = Field(default_factory=dict)
    id_to_token: Dict[int, str] = Field(default_factory=dict)

    def setup_prompt(self, user_prompt: str, model: Any) -> None:
        """Sets up and encodes the prompt template boundaries for tracking.

        Args:
            user_prompt (str): The user's query prompt.
            model (Any): The model engine instance.
        """
        self.user_prompt = user_prompt
        self.model = model
        self.prefix = '{"prompt":' + json.dumps(user_prompt) + ',"name":"'
        self.prefix_tokens = model.encode(self.prefix).flatten().tolist()
        self.decoded_prefix = model.decode(self.prefix_tokens)
        self.fn_suffix_tokens = {}

        for fn_name in self.func_def.func_def_dict:
            full_str = self.prefix + fn_name + self.param_suffix
            full_tokens = model.encode(full_str).flatten().tolist()
            self.fn_suffix_tokens[fn_name] = full_tokens[
                len(self.prefix_tokens):
            ]

    def mask(
        self,
        generated_text: str,
        logits: List[float],
        id_to_token: Dict[int, str]
    ) -> List[float]:
        """Applies logits constraint mask using current tracking state.

        Args:
            generated_text (str): The currently generated string slice.
            logits (List[float]): The raw logits array.
            id_to_token (Dict[int, str]): Token ID to vocabulary dictionary.

        Returns:
            List[float]: The constrained logits array.
        """
        if not self.user_prompt or self.model is None or not self.prefix:
            return logits

        self.id_to_token.update(id_to_token)
        allowed_ids = self._get_allowed_token_ids(generated_text)

        if allowed_ids:
            new_logits = np.full_like(logits, -float("inf"))
            for token_id in allowed_ids:
                new_logits[token_id] = logits[token_id]
            return cast(List[float], new_logits.tolist())

        return logits

    @staticmethod
    def end_condition(current_text: str) -> bool:
        """Evaluates if the text forms a complete parseable JSON payload.

        Args:
            current_text (str): The generated JSON text.

        Returns:
            bool: True if the text parses successfully as JSON,
                False otherwise.
        """
        try:
            json.loads(current_text)
            return True
        except json.JSONDecodeError:
            return False

    def _build_rem_prefix(self, generated_text: str) -> List[int]:
        """Identifies allowed tokens that can complete the remaining prefix.

        Args:
            generated_text (str): The generated text string.

        Returns:
            List[int]: The list of allowed token IDs.
        """
        allowed_ids: List[int] = []
        rem_prefix = self.decoded_prefix[len(generated_text):]
        for token_id, token_str in self.id_to_token.items():
            if rem_prefix.startswith(token_str):
                allowed_ids.append(token_id)
            elif token_str.startswith(rem_prefix):
                extra = token_str[len(rem_prefix):]
                for fn_name in self.func_def.func_def_dict:
                    target = fn_name + self.param_suffix
                    if target.startswith(extra) or extra.startswith(target):
                        allowed_ids.append(token_id)
                        break
        return allowed_ids

    def _define_name_phase(self, rem: str) -> Tuple[bool, Any]:
        """Checks if the decoder is currently completing a function name.

        Args:
            rem (str): The remaining generated text slice.

        Returns:
            Tuple[bool, Any]: (True if in name phase, the active function name
                string if fully matched).
        """
        is_in_name_phase: bool = False
        active_fn: Optional[str] = None
        for fn_name in self.func_def.func_def_dict:
            target = fn_name + self.param_suffix
            if rem == target:
                active_fn = fn_name
                break
            elif target.startswith(rem):
                is_in_name_phase = True
                break

        return (is_in_name_phase, active_fn)

    def _resolve_name_phase(self, rem: str) -> List[int]:
        """Retrieves tokens to continue completing the function name prefix.

        Args:
            rem (str): The remaining generated text slice.

        Returns:
            List[int]: The list of allowed token IDs.
        """
        allowed_ids: List[int] = []
        for token_id, token_str in self.id_to_token.items():
            for fn_name in self.func_def.func_def_dict:
                target = fn_name + self.param_suffix
                if target.startswith(rem + token_str):
                    allowed_ids.append(token_id)
                    break
                elif (rem + token_str).startswith(target):
                    extra = (rem + token_str)[len(target):]
                    if extra == "" or extra == '"':
                        allowed_ids.append(token_id)
                        break
                    elif extra.startswith('"'):
                        key_part = extra[1:]
                        for key in (
                            self.func_def.func_def_dict[fn_name].parameters
                        ):
                            if (
                                (key + ":").startswith(key_part)
                                or key_part.startswith(key + ":")
                            ):
                                allowed_ids.append(token_id)
                                break
        return allowed_ids

    def _get_allowed_token_ids(
        self,
        generated_text: str,
    ) -> List[int]:
        """Determines allowed token IDs based on current generated text state.

        Args:
            generated_text (str): The complete text generated so far.

        Returns:
            List[int]: List of allowed token IDs for the next step.
        """
        if len(generated_text) < len(self.decoded_prefix):
            return self._build_rem_prefix(generated_text)

        allowed_ids: List[int] = []
        rem = generated_text[len(self.decoded_prefix):]
        is_in_name_phase, active_fn = self._define_name_phase(rem)

        if is_in_name_phase:
            return self._resolve_name_phase(rem)

        if active_fn is None:
            for fn_name in self.func_def.func_def_dict:
                target = fn_name + self.param_suffix
                if rem.startswith(target):
                    active_fn = fn_name
                    break
            return []

        func_schema = self.func_def.func_def_dict[active_fn]
        param_text = rem[len(active_fn) + len(self.param_suffix):]
        scan_res = self._scan_parameters(param_text, func_schema)
        state: Optional[JSONState] = scan_res.get("state")
        remaining_keys: Set[str] = scan_res.get("remaining_keys") or set()
        partial_key: str = scan_res.get("partial_key") or ""
        current_key: Optional[str] = scan_res.get("current_key")
        param_type: str = scan_res.get("type") or ""
        partial_value: str = scan_res.get("partial_value") or ""
        remaining_text: str = scan_res.get("remaining_text") or ""

        if state == JSONState.KEY_START:
            for token_id, token_str in self.id_to_token.items():
                for K in remaining_keys:
                    target = '"' + K + '":'
                    if target.startswith(token_str):
                        allowed_ids.append(token_id)
                        break
                    elif token_str.startswith(target):
                        extra = token_str[len(target):]
                        if self._is_valid_value_prefix(
                            extra,
                            func_schema.parameters[K].type,
                            remaining_keys,
                            K
                        ):
                            allowed_ids.append(token_id)
                            break

        elif state == JSONState.KEY_PARTIAL:
            for token_id, token_str in self.id_to_token.items():
                for K in remaining_keys:
                    if K.startswith(partial_key):
                        target = (K + '":')[len(partial_key):]
                        if target.startswith(token_str):
                            allowed_ids.append(token_id)
                            break
                        elif token_str.startswith(target):
                            extra = token_str[len(target):]
                            if self._is_valid_value_prefix(
                                extra,
                                func_schema.parameters[K].type,
                                remaining_keys,
                                K
                            ):
                                allowed_ids.append(token_id)
                                break

        elif state == JSONState.COLON_START:
            if current_key is None:
                return []
            for token_id, token_str in self.id_to_token.items():
                if ":".startswith(token_str):
                    allowed_ids.append(token_id)
                elif token_str.startswith(":"):
                    extra = token_str[1:]
                    if self._is_valid_value_prefix(
                        extra,
                        func_schema.parameters[current_key].type,
                        remaining_keys,
                        current_key
                    ):
                        allowed_ids.append(token_id)

        elif state == JSONState.VALUE_START:
            if current_key is None:
                return []
            for token_id, token_str in self.id_to_token.items():
                if self._is_valid_value_prefix(
                    token_str,
                    param_type,
                    remaining_keys,
                    current_key
                ):
                    allowed_ids.append(token_id)

        elif state == JSONState.VALUE_PARTIAL:
            if current_key is None:
                return []
            if param_type == 'string':
                for token_id, token_str in self.id_to_token.items():
                    clean_token = token_str.replace('Ġ', '').lstrip()
                    is_regex_start = (
                        clean_token.startswith('.')
                        or clean_token.startswith('*')
                    )
                    if (
                        current_key == "regex"
                        and partial_value == ""
                        and is_regex_start
                    ):
                        continue
                    idx = find_first_unescaped_quote(token_str)
                    if idx == -1:
                        allowed_ids.append(token_id)
                    else:
                        extra = token_str[idx+1:]
                        if self._is_valid_separator_prefix(
                            extra, remaining_keys
                        ):
                            allowed_ids.append(token_id)

            elif param_type == 'number':
                for token_id, token_str in self.id_to_token.items():
                    if number_regex.match(partial_value + token_str):
                        allowed_ids.append(token_id)
                    elif is_complete_number(partial_value):
                        if self._is_valid_separator_prefix(
                            token_str, remaining_keys
                        ):
                            allowed_ids.append(token_id)

            elif param_type in ('boolean', 'bool'):
                target = 'true' if partial_value.startswith('t') else 'false'
                for token_id, token_str in self.id_to_token.items():
                    if target.startswith(partial_value + token_str):
                        allowed_ids.append(token_id)
                    elif (target == partial_value + token_str or
                          (partial_value + token_str).startswith(target)):
                        extra = (partial_value + token_str)[len(target):]
                        if self._is_valid_separator_prefix(
                            extra, remaining_keys
                        ):
                            allowed_ids.append(token_id)

        elif state == JSONState.COMMA_OR_CLOSE_START:
            for token_id, token_str in self.id_to_token.items():
                if self._is_valid_separator_prefix(
                    token_str, remaining_keys
                ):
                    allowed_ids.append(token_id)

        elif state == JSONState.CLOSED:
            if remaining_text == "":
                for token_id, token_str in self.id_to_token.items():
                    if '}'.startswith(token_str) or token_str.startswith('}'):
                        allowed_ids.append(token_id)
            elif remaining_text == "}":
                allowed_ids.append(EOS_TOKEN_ID)

        return allowed_ids

    def _scan_parameters(
        self,
        param_text: str,
        func_schema: FunctionSchema
    ) -> Dict[str, Any]:
        """Scans generated parameters string to parse the state of the JSON.

        Args:
            param_text (str): The parameter substring generated so far.
            func_schema (FunctionSchema): The schema of the active function.

        Returns:
            Dict[str, Any]: A dictionary containing parsing state metrics.
        """
        remaining_keys: Set[str] = set(func_schema.parameters.keys())
        idx: int = 0
        n: int = len(param_text)
        current_key: Optional[str] = None
        state: JSONState = JSONState.KEY

        while idx < n:
            if state == JSONState.KEY:
                if param_text[idx] == '"':
                    close_idx = param_text.find('"', idx + 1)
                    if close_idx == -1:
                        partial_key = param_text[idx+1:]
                        return {
                            'state': JSONState.KEY_PARTIAL,
                            'remaining_keys': remaining_keys,
                            'partial_key': partial_key
                        }
                    else:
                        key = param_text[idx+1:close_idx]
                        if key in remaining_keys:
                            current_key = key
                            remaining_keys.remove(key)
                        idx = close_idx + 1
                        state = JSONState.COLON
                else:
                    idx += 1
            elif state == JSONState.COLON:
                if param_text[idx] == ':':
                    state = JSONState.VALUE
                    idx += 1
                else:
                    idx += 1
            elif state == JSONState.VALUE:
                if current_key is None:
                    return {'state': JSONState.ERROR}
                param_type = func_schema.parameters[current_key].type
                if param_type == 'string':
                    if param_text[idx] == '"':
                        close_idx = -1
                        i = idx + 1
                        while i < n:
                            if param_text[i] == '"':
                                if not is_quote_escaped(param_text, i):
                                    close_idx = i
                                    break
                            i += 1
                        if close_idx == -1:
                            partial_value = param_text[idx+1:]
                            return {
                                'state': JSONState.VALUE_PARTIAL,
                                'remaining_keys': remaining_keys,
                                'current_key': current_key,
                                'partial_value': partial_value,
                                'type': 'string'
                            }
                        else:
                            idx = close_idx + 1
                            state = JSONState.COMMA_OR_CLOSE
                    else:
                        idx += 1
                elif param_type == 'number':
                    start_idx = idx
                    while idx < n and param_text[idx] in '-+.eE0123456789':
                        idx += 1
                    num_str = param_text[start_idx:idx]
                    if idx < n:
                        state = JSONState.COMMA_OR_CLOSE
                    else:
                        return {
                            'state': JSONState.VALUE_PARTIAL,
                            'remaining_keys': remaining_keys,
                            'current_key': current_key,
                            'partial_value': num_str,
                            'type': 'number'
                        }
                elif param_type in ('boolean', 'bool'):
                    if param_text[idx:].startswith('true'):
                        idx += 4
                        state = JSONState.COMMA_OR_CLOSE
                    elif param_text[idx:].startswith('false'):
                        idx += 5
                        state = JSONState.COMMA_OR_CLOSE
                    else:
                        partial = param_text[idx:]
                        return {
                            'state': JSONState.VALUE_PARTIAL,
                            'remaining_keys': remaining_keys,
                            'current_key': current_key,
                            'partial_value': partial,
                            'type': 'boolean'
                        }
            elif state == JSONState.COMMA_OR_CLOSE:
                if param_text[idx] == ',':
                    state = JSONState.KEY
                    idx += 1
                elif param_text[idx] == '}':
                    idx += 1
                    return {
                        'state': JSONState.CLOSED,
                        'remaining_text': param_text[idx:]
                    }
                else:
                    idx += 1

        if state == JSONState.KEY:
            return {
                'state': JSONState.KEY_START,
                'remaining_keys': remaining_keys
            }
        elif state == JSONState.COLON:
            return {
                'state': JSONState.COLON_START,
                'remaining_keys': remaining_keys,
                'current_key': current_key
            }
        elif state == JSONState.VALUE:
            if current_key is None:
                return {'state': JSONState.ERROR}
            param_type = func_schema.parameters[current_key].type
            return {
                'state': JSONState.VALUE_START,
                'remaining_keys': remaining_keys,
                'current_key': current_key,
                'type': param_type
            }
        elif state == JSONState.COMMA_OR_CLOSE:
            return {
                'state': JSONState.COMMA_OR_CLOSE_START,
                'remaining_keys': remaining_keys
            }
        return {'state': JSONState.ERROR}

    def _is_valid_value_prefix(
        self,
        extra: str,
        param_type: str,
        remaining_keys: Set[str],
        current_key: Optional[str] = None
    ) -> bool:
        """Checks if a suffix string matches a valid prefix for values.

        Args:
            extra (str): The suffix string to evaluate.
            param_type (str): The expected parameter type.
            remaining_keys (Set[str]): Set of remaining parameter keys.
            current_key (Optional[str]): The parameter key currently processed.

        Returns:
            bool: True if the suffix is a valid value prefix representation,
                False otherwise.
        """
        if param_type == "string":
            if '"'.startswith(extra):
                return True
            if extra.startswith('"'):
                val_part = extra[1:]
                is_reg_pre = (
                    val_part.startswith('.')
                    or val_part.startswith('*')
                )
                if current_key == "regex" and is_reg_pre:
                    return False
                idx = find_first_unescaped_quote(val_part)
                if idx == -1:
                    return True
                else:
                    sep_part = val_part[idx+1:]
                    return self._is_valid_separator_prefix(
                        sep_part, remaining_keys
                    )
            return False

        elif param_type == "number":
            if number_regex.match(extra):
                return True

            for i in range(len(extra)):
                if extra[i] in ',}':
                    num_part = extra[:i]
                    sep_part = extra[i:]
                    if (is_complete_number(num_part) and
                        self._is_valid_separator_prefix(
                            sep_part, remaining_keys)):
                        return True
            return False

        elif param_type in ("boolean", "bool"):
            for target in ("true", "false"):
                if target.startswith(extra):
                    return True
                if extra.startswith(target):
                    sep_part = extra[len(target):]
                    if self._is_valid_separator_prefix(
                        sep_part, remaining_keys
                    ):
                        return True

            return False
        return False

    def _is_valid_separator_prefix(
        self,
        extra: str,
        remaining_keys: Set[str]
    ) -> bool:
        """Checks if a suffix matches a valid separator or closing structure.

        Args:
            extra (str): The suffix string to evaluate.
            remaining_keys (Set[str]): Set of remaining parameter keys.

        Returns:
            bool: True if the suffix matches a valid separator,
                False otherwise.
        """
        if not remaining_keys:
            return "}}".startswith(extra) or extra.startswith("}}")

        if ",".startswith(extra):
            return True

        elif extra.startswith(","):
            next_extra = extra[1:]
            if '"'.startswith(next_extra):
                return True

            if next_extra.startswith('"'):
                key_part = next_extra[1:]
                for K in remaining_keys:
                    is_match = (
                        (K + '":').startswith(key_part)
                        or key_part.startswith(K + '":')
                    )
                    if is_match:
                        return True

        return False


class ModelEngine(BaseModel):
    """Execution engine wrapping LLM and managing tokenizer configurations.

    Attributes:
        model_name (str): Model weights name. Defaults to Qwen/Qwen3-0.6B.
        vocab_dict (Dict[str, int]): Map of vocabulary tokens to IDs.
        id_to_token (Dict[int, str]): Reverse lookup of token IDs to strings.
        id_to_token_decoded (Dict[int, str]): Token strings with mappings.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = "Qwen/Qwen3-0.6B"
    vocab_dict: Dict[str, int] = Field(default_factory=dict)
    id_to_token: Dict[int, str] = Field(default_factory=dict)
    id_to_token_decoded: Dict[int, str] = Field(default_factory=dict)
    _model: Small_LLM_Model = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        """Loads model weights, tokenizer file, and maps vocabulary."""
        self._model = Small_LLM_Model(model_name=self.model_name)

        if os.path.exists(TOKENIZER_PATH):
            with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)

        else:
            tokenizer = self._model.get_path_to_tokenizer_file()
            with open(tokenizer, "r", encoding="utf-8") as f:
                data = json.load(f)

            os.makedirs(os.path.dirname(TOKENIZER_PATH), exist_ok=True)
            with open(TOKENIZER_PATH, "w") as f:
                json.dump(data, f, indent=2)

        self.vocab_dict.update(data["model"]["vocab"])
        self.id_to_token.update({
            v: k for k, v in self.vocab_dict.items()
        })
        self.id_to_token_decoded.update({
            v: k.replace('Ġ', ' ').replace('Ċ', '\n')
            for k, v in self.vocab_dict.items()
        })

    def encode(self, text: str) -> Any:
        """Encodes input string to tensor token IDs.

        Args:
            text (str): The input text message.

        Returns:
            Any: Tokenized input IDs.
        """
        return self._model.encode(text)

    def decode(self, ids: Any) -> str:
        """Decodes token IDs back into string text.

        Args:
            ids (Any): The token IDs.

        Returns:
            str: The decoded string representation.
        """
        return self._model.decode(ids)

    def get_logits_from_input_ids(self, input_ids: List[int]) -> List[float]:
        """Retrieves logits distribution from model based on input sequence.

        Args:
            input_ids (List[int]): List of sequence token IDs.

        Returns:
            List[float]: The logits distribution.
        """
        return self._model.get_logits_from_input_ids(input_ids)

    def generate(
        self,
        tracker: Optional[TokenizerMaskProtocol],
        prompt: str,
        base_prompt_str: Optional[str] = None,
        max_new_tokens: int = 50,
        answer_str: str = "ANSWER: ",
        printable: bool = False
    ) -> str:
        """Autoregressive token generation using tracker constraints.

        Args:
            tracker (Optional[TokenizerMaskProtocol]): The constraint tracker.
            prompt (str): The query prompt.
            base_prompt_str (Optional[str]): System prompt base template.
            max_new_tokens (int): The maximum tokens limit. Defaults to 50.
            answer_str (str): Visual logging prefix. Defaults to "ANSWER: ".
            printable (bool): If True, prints tokens. Defaults to False.

        Returns:
            str: The generated text response.
        """
        prompt_tokens = self.encode(prompt).flatten().tolist()
        len_prompt = len(prompt_tokens)
        tokens_list = prompt_tokens.copy()

        for _ in range(max_new_tokens):

            generated_tokens = tokens_list[len_prompt:]
            generated_text = self.decode(generated_tokens)
            logits = self.get_logits_from_input_ids(tokens_list)

            try:
                if tracker:
                    logits = tracker.mask(
                        generated_text=generated_text,
                        logits=logits,
                        id_to_token=self.id_to_token_decoded
                    )

                next_token = int(np.argmax(logits))
                if next_token == EOS_TOKEN_ID:
                    break
                tokens_list.append(next_token)
                current_text = self.decode(tokens_list[len_prompt:])

                if printable:
                    print(f"\r{answer_str}{current_text}", end="", flush=True)

                if tracker and tracker.end_condition(current_text):
                    break

            except Exception as e:
                print_error(
                    error_msg=f"Model generation error: {e}",
                    critical=False
                )
                break

        return self.decode(tokens_list[len_prompt:])

    def test_model(
        self,
        tracker: Optional[TokenizerMaskProtocol],
        prompt: BasePrompt,
        test: str,
        test_num: int = 0,
        max_new_tokens: int = 50
    ) -> str:
        """Runs generation test for a single query prompt.

        Args:
            tracker (Optional[TokenizerMaskProtocol]): The constraint tracker.
            prompt (BasePrompt): The formatted system prompt manager.
            test (str): The user query.
            test_num (int): Visual indicator test identifier. Defaults to 0.
            max_new_tokens (int): The token limit. Defaults to 50.

        Returns:
            str: The generated response.
        """
        print("\n" + f" GENERATION_TEST_{test_num:02d} ".center(UX_WIDTH, "="))
        print("PROMPT: ", test)
        composed_prompt: str = prompt.compose_base_prompt(test)
        if tracker and hasattr(tracker, "setup_prompt"):
            tracker.setup_prompt(test, self)
        generated_text = self.generate(
            tracker=tracker,
            prompt=composed_prompt,
            max_new_tokens=max_new_tokens,
            printable=True
        )

        return generated_text


def main() -> None:
    """CLI entrypoint loading resources and executing predictions."""
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument("--functions_definition", default=FUNC_DEF_PATH)
    parser.add_argument("--input", default=FUNC_CALL_TESTS_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)
    arg = parser.parse_args()

    try:
        welcome()
        section_header("LOADING IO FILES")
        func_def = DynamicFunctionDefinitions(
            func_def_path=arg.functions_definition
        )
        print(f"• Loading function schemas from {arg.functions_definition}...")
        user_prompts = UserPrompts(test_prompts_path=arg.input)
        print(f"• Loading test prompts from {arg.input}...")
        print(f"• Output file will be saved to {arg.output}...\n")
        sleep(1)
        section_header("LOADING MODEL")
        model = ModelEngine()
        print()
        sleep(1)
        section_header("LAUNCHING TESTS")
        json_traker = ConstrainedJSONTracker(func_def=func_def)

        model_response_list: List[Dict[str, Any]] = []

        for i, test in enumerate(user_prompts.get()):
            prompt = BasePrompt(
                func_def_dict=func_def.func_def_dict,
            )
            generated_text = model.test_model(
                prompt=prompt,
                tracker=json_traker,
                test=test,
                test_num=i,
                max_new_tokens=100
            )

            try:
                dict_response = json.loads(generated_text)
                fn_name = dict_response.get("name")
                if not fn_name or fn_name not in func_def.validators:
                    raise ValueError(
                        ERRORS["unknown_fn_error"].format(fn_name=fn_name)
                    )

                validated_params = func_def.validators[fn_name](
                    **dict_response.get("parameters", {})
                )
                dict_response["parameters"] = validated_params.model_dump()
                model_response_list.append(dict_response)

            except json.JSONDecodeError as e:
                print_error(
                    error_msg=ERRORS["json_decode_error"].format(error=e),
                    critical=False
                )

            except Exception as e:
                print_error(
                    error_msg=ERRORS["validation_error"].format(error=e),
                    critical=False
                )

        compose_output_file(model_response_list)
        goodbye()

    except KeyboardInterrupt:
        goodbye()


if __name__ == "__main__":
    main()

from llm_sdk.llm_sdk import Small_LLM_Model
from typing import Union, Optional, Dict, List, Tuple, Any, Protocol
from pydantic import BaseModel, create_model
from src.utils import TOKENIZER_PATH, BASE_PROMPT_PATH, FUNC_DEF_PATH
from src.utils import OUTPUT_PATH, FUNC_CALL_TESTS_PATH, EOS_TOKEN_ID
from src.utils import compose_output_file, debug_output_token_list
from src import __description__
import numpy as np
import json
import argparse
import os
import re
import sys

# Regex and helpers for parsing partial JSON values
number_prefix_regex = re.compile(r'^[+-]?[0-9]*\.?[0-9]*([eE][+-]?[0-9]*)?$')

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


class ParameterInfo(BaseModel):
    """
    """
    type: str


class FunctionCallResult(BaseModel):
    """
    """
    prompt: str
    name: str
    parameters: BaseModel


class FunctionSchema(BaseModel):
    """
    """
    name: str
    description: str
    parameters: Dict[str, ParameterInfo]
    returns: Dict[str, str]


class DynamicFunctionDefinitions:
    """
    """
    def __init__(
            self,
            func_def_path: str) -> None:
        self._func_def_path: str = func_def_path
        self.func_def_dict: Dict[str, FunctionSchema] = {}
        self.validators: Dict[str, BaseModel] = {}
        self._load_functions_definition()
        # Add fn_unknown schema here so it's registered, validated, and used in decoding
        self.func_def_dict["fn_unknown"] = FunctionSchema(
            name="fn_unknown",
            description="Call this function strictly when the user's request cannot be fulfilled by any other available function in the catalog.",
            parameters={"reason": ParameterInfo(type="string")},
            returns={"type": "string"}
        )
        self._preconfigure_validators()

    def _load_functions_definition(self) -> None:
        """
        """
        try:
            with open(self._func_def_path, "r") as file:
                raw = json.load(file)
                func_def_list: List[FunctionSchema] = [
                    FunctionSchema(**f) for f in raw
                    ]
                self.func_def_dict.update({
                    func.name: func for func in func_def_list
                })

        except FileNotFoundError:
            print(f"[ERROR] Function definitions file not found: {self._func_def_path}")
            sys.exit(1)

        except json.JSONDecodeError:
            print(f"[ERROR] Failed to parse JSON in function definitions: {self._func_def_path}")
            sys.exit(1)

    def _preconfigure_validators(self) -> None:
        """
        """
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


class UserPrompts:
    """
    """
    def __init__(
            self,
            test_prompts_path: str
            ) -> None:
        self.test_prompts_path: str = test_prompts_path
        self._prompt_list: List[str] = []
        self._load_user_prompts()

    def _load_user_prompts(self) -> None:
        """
        """
        try:
            with open(self.test_prompts_path, "r") as file:
                raw: List[Dict[str, str]] = json.load(file)
                self._prompt_list.extend([p["prompt"] for p in raw])

        except FileNotFoundError:
            print(f"[ERROR] User prompts file not found: {self.test_prompts_path}")
            sys.exit(1)

        except json.JSONDecodeError:
            print(f"[ERROR] Failed to parse JSON in user prompts: {self.test_prompts_path}")
            sys.exit(1)

    def get(self) -> List[str]:
        """
        """
        return self._prompt_list


class TokenizerMaskProtocol(Protocol):
    def mask(
        self,
        generated_text: str,
        logits: List[float],
        id_to_token: Dict[int, str]
        ) -> List[float]:
        """
        """
        ...


class JSONStateTracker:
    """
    State-machine based JSON state tracker for constrained decoding.
    """
    def __init__(
            self,
            func_def: DynamicFunctionDefinitions
        ) -> None:
        self.func_def: DynamicFunctionDefinitions = func_def
        self.user_prompt: Optional[str] = None
        self.model: Optional[Any] = None
        self.prefix: Optional[str] = None
        self.prefix_tokens: List[int] = []
        self.param_suffix: str = '","parameters":{'
        self.fn_suffix_tokens: Dict[str, List[int]] = {}

    def setup_prompt(self, user_prompt: str, model: Any) -> None:
        self.user_prompt = user_prompt
        self.model = model
        
        # Precompute prefix and its tokens
        self.prefix = '{"prompt":' + json.dumps(user_prompt) + ',"name":"'
        self.prefix_tokens = model.encode(self.prefix).flatten().tolist()
        
        # Precompute candidate function name suffix tokens (name + param_suffix)
        self.fn_suffix_tokens = {}
        for fn_name in self.func_def.func_def_dict:
            full_str = self.prefix + fn_name + self.param_suffix
            full_tokens = model.encode(full_str).flatten().tolist()
            # Extract suffix tokens starting after prefix
            self.fn_suffix_tokens[fn_name] = full_tokens[len(self.prefix_tokens):]

    def mask(
            self,
            generated_text: str,
            logits: List[float],
            id_to_token: Dict[int, str]
            ) -> List[float]:
        """
        """
        if self.user_prompt is None or self.model is None or self.prefix is None:
            return logits

        allowed_ids = self._get_allowed_token_ids(generated_text, id_to_token)

        if allowed_ids:
            new_logits = np.full_like(logits, -float("inf"))
            for token_id in allowed_ids:
                new_logits[token_id] = logits[token_id]
            return new_logits.tolist()
            
        return logits

    @staticmethod
    def end_condition(current_text: str) -> bool:
        """
        """
        try:
            json.loads(current_text)
            return True
        except json.JSONDecodeError:
            return False

    def _get_allowed_token_ids(self, generated_text: str, id_to_token: Dict[int, str]) -> List[int]:
        if len(generated_text) < len(self.prefix):
            rem_prefix = self.prefix[len(generated_text):]
            allowed_ids = []
            for token_id, token_str in id_to_token.items():
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

        rem = generated_text[len(self.prefix):]
        is_in_name_phase = True
        active_fn = None
        for fn_name in self.func_def.func_def_dict:
            target = fn_name + self.param_suffix
            if rem == target:
                is_in_name_phase = False
                active_fn = fn_name
                break
            elif target.startswith(rem):
                is_in_name_phase = True
                break
        else:
            is_in_name_phase = False

        if is_in_name_phase:
            allowed_ids = []
            for token_id, token_str in id_to_token.items():
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
                            for key in self.func_def.func_def_dict[fn_name].parameters:
                                if (key + '":').startswith(key_part) or key_part.startswith(key + '":'):
                                    allowed_ids.append(token_id)
                                    break
                            else:
                                continue
                            break
            return allowed_ids

        if active_fn is None:
            for fn_name in self.func_def.func_def_dict:
                target = fn_name + self.param_suffix
                if rem.startswith(target):
                    active_fn = fn_name
                    break

        if active_fn is None:
            return []

        func_schema = self.func_def.func_def_dict[active_fn]
        param_text = rem[len(active_fn) + len(self.param_suffix):]
        scan_res = self._scan_parameters(param_text, func_schema)
        state = scan_res['state']

        allowed_ids = []
        
        if state == 'KEY_START':
            remaining_keys = scan_res['remaining_keys']
            for token_id, token_str in id_to_token.items():
                for K in remaining_keys:
                    target = '"' + K + '":'
                    if target.startswith(token_str):
                        allowed_ids.append(token_id)
                        break
                    elif token_str.startswith(target):
                        extra = token_str[len(target):]
                        if self._is_valid_value_prefix(extra, func_schema.parameters[K].type, remaining_keys, func_schema):
                            allowed_ids.append(token_id)
                            break
                            
        elif state == 'KEY_PARTIAL':
            remaining_keys = scan_res['remaining_keys']
            partial_key = scan_res['partial_key']
            for token_id, token_str in id_to_token.items():
                for K in remaining_keys:
                    if K.startswith(partial_key):
                        target = (K + '":')[len(partial_key):]
                        if target.startswith(token_str):
                            allowed_ids.append(token_id)
                            break
                        elif token_str.startswith(target):
                            extra = token_str[len(target):]
                            if self._is_valid_value_prefix(extra, func_schema.parameters[K].type, remaining_keys, func_schema):
                                allowed_ids.append(token_id)
                                break

        elif state == 'COLON_START':
            remaining_keys = scan_res['remaining_keys']
            current_key = scan_res['current_key']
            for token_id, token_str in id_to_token.items():
                if ':'.startswith(token_str):
                    allowed_ids.append(token_id)
                elif token_str.startswith(':'):
                    extra = token_str[1:]
                    if self._is_valid_value_prefix(extra, func_schema.parameters[current_key].type, remaining_keys, func_schema):
                        allowed_ids.append(token_id)

        elif state == 'VALUE_START':
            remaining_keys = scan_res['remaining_keys']
            current_key = scan_res['current_key']
            param_type = scan_res['type']
            for token_id, token_str in id_to_token.items():
                if self._is_valid_value_prefix(token_str, param_type, remaining_keys, func_schema):
                    allowed_ids.append(token_id)

        elif state == 'VALUE_PARTIAL':
            remaining_keys = scan_res['remaining_keys']
            current_key = scan_res['current_key']
            partial_value = scan_res['partial_value']
            param_type = scan_res['type']
            
            if param_type == 'string':
                for token_id, token_str in id_to_token.items():
                    idx = find_first_unescaped_quote(token_str)
                    if idx == -1:
                        allowed_ids.append(token_id)
                    else:
                        extra = token_str[idx+1:]
                        if self._is_valid_separator_prefix(extra, remaining_keys, func_schema):
                            allowed_ids.append(token_id)
                            
            elif param_type == 'number':
                for token_id, token_str in id_to_token.items():
                    if number_prefix_regex.match(partial_value + token_str):
                        allowed_ids.append(token_id)
                    elif is_complete_number(partial_value):
                        if self._is_valid_separator_prefix(token_str, remaining_keys, func_schema):
                            allowed_ids.append(token_id)
                            
            elif param_type in ('boolean', 'bool'):
                target = 'true' if partial_value.startswith('t') else 'false'
                for token_id, token_str in id_to_token.items():
                    if target.startswith(partial_value + token_str):
                        allowed_ids.append(token_id)
                    elif target == partial_value + token_str or (partial_value + token_str).startswith(target):
                        extra = (partial_value + token_str)[len(target):]
                        if self._is_valid_separator_prefix(extra, remaining_keys, func_schema):
                            allowed_ids.append(token_id)

        elif state == 'COMMA_OR_CLOSE_START':
            remaining_keys = scan_res['remaining_keys']
            for token_id, token_str in id_to_token.items():
                if self._is_valid_separator_prefix(token_str, remaining_keys, func_schema):
                    allowed_ids.append(token_id)

        elif state == 'CLOSED':
            remaining_text = scan_res['remaining_text']
            if remaining_text == "":
                for token_id, token_str in id_to_token.items():
                    if '}'.startswith(token_str) or token_str.startswith('}'):
                        allowed_ids.append(token_id)
            elif remaining_text == "}":
                allowed_ids.append(EOS_TOKEN_ID)

        return allowed_ids

    def _scan_parameters(self, param_text: str, func_schema: FunctionSchema) -> Dict[str, Any]:
        remaining_keys = set(func_schema.parameters.keys())
        idx = 0
        n = len(param_text)
        current_key = None
        state = 'KEY'
        
        while idx < n:
            if state == 'KEY':
                if param_text[idx] == '"':
                    close_idx = param_text.find('"', idx + 1)
                    if close_idx == -1:
                        partial_key = param_text[idx+1:]
                        return {
                            'state': 'KEY_PARTIAL',
                            'remaining_keys': remaining_keys,
                            'partial_key': partial_key
                        }
                    else:
                        key = param_text[idx+1:close_idx]
                        if key in remaining_keys:
                            current_key = key
                            remaining_keys.remove(key)
                        idx = close_idx + 1
                        state = 'COLON'
                else:
                    idx += 1
            elif state == 'COLON':
                if param_text[idx] == ':':
                    state = 'VALUE'
                    idx += 1
                else:
                    idx += 1
            elif state == 'VALUE':
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
                                'state': 'VALUE_PARTIAL',
                                'remaining_keys': remaining_keys,
                                'current_key': current_key,
                                'partial_value': partial_value,
                                'type': 'string'
                            }
                        else:
                            idx = close_idx + 1
                            state = 'COMMA_OR_CLOSE'
                    else:
                        idx += 1
                elif param_type == 'number':
                    start_idx = idx
                    while idx < n and param_text[idx] in '-+.eE0123456789':
                        idx += 1
                    num_str = param_text[start_idx:idx]
                    if idx < n:
                        state = 'COMMA_OR_CLOSE'
                    else:
                        return {
                            'state': 'VALUE_PARTIAL',
                            'remaining_keys': remaining_keys,
                            'current_key': current_key,
                            'partial_value': num_str,
                            'type': 'number'
                        }
                elif param_type in ('boolean', 'bool'):
                    if param_text[idx:].startswith('true'):
                        idx += 4
                        state = 'COMMA_OR_CLOSE'
                    elif param_text[idx:].startswith('false'):
                        idx += 5
                        state = 'COMMA_OR_CLOSE'
                    else:
                        partial = param_text[idx:]
                        return {
                            'state': 'VALUE_PARTIAL',
                            'remaining_keys': remaining_keys,
                            'current_key': current_key,
                            'partial_value': partial,
                            'type': 'boolean'
                        }
            elif state == 'COMMA_OR_CLOSE':
                if param_text[idx] == ',':
                    state = 'KEY'
                    idx += 1
                elif param_text[idx] == '}':
                    idx += 1
                    return {
                        'state': 'CLOSED',
                        'remaining_text': param_text[idx:]
                    }
                else:
                    idx += 1
                    
        if state == 'KEY':
            return {
                'state': 'KEY_START',
                'remaining_keys': remaining_keys
            }
        elif state == 'COLON':
            return {
                'state': 'COLON_START',
                'remaining_keys': remaining_keys,
                'current_key': current_key
            }
        elif state == 'VALUE':
            param_type = func_schema.parameters[current_key].type
            return {
                'state': 'VALUE_START',
                'remaining_keys': remaining_keys,
                'current_key': current_key,
                'type': param_type
            }
        elif state == 'COMMA_OR_CLOSE':
            return {
                'state': 'COMMA_OR_CLOSE_START',
                'remaining_keys': remaining_keys
            }
        return {'state': 'ERROR'}

    def _is_valid_value_prefix(self, extra: str, param_type: str, remaining_keys: set, func_schema: FunctionSchema) -> bool:
        if param_type == 'string':
            if '"'.startswith(extra):
                return True
            if extra.startswith('"'):
                val_part = extra[1:]
                idx = find_first_unescaped_quote(val_part)
                if idx == -1:
                    return True
                else:
                    sep_part = val_part[idx+1:]
                    return self._is_valid_separator_prefix(sep_part, remaining_keys, func_schema)
            return False
        elif param_type == 'number':
            if number_prefix_regex.match(extra):
                return True
            for i in range(len(extra)):
                if extra[i] in ',}':
                    num_part = extra[:i]
                    sep_part = extra[i:]
                    if is_complete_number(num_part) and self._is_valid_separator_prefix(sep_part, remaining_keys, func_schema):
                        return True
            return False
        elif param_type in ('boolean', 'bool'):
            for target in ('true', 'false'):
                if target.startswith(extra):
                    return True
                if extra.startswith(target):
                    sep_part = extra[len(target):]
                    if self._is_valid_separator_prefix(sep_part, remaining_keys, func_schema):
                        return True
            return False
        return False

    def _is_valid_separator_prefix(self, extra: str, remaining_keys: set, func_schema: FunctionSchema) -> bool:
        if not remaining_keys:
            return '}}'.startswith(extra) or extra.startswith('}}')
        else:
            if ','.startswith(extra):
                return True
            if extra.startswith(','):
                next_extra = extra[1:]
                if '"'.startswith(next_extra):
                    return True
                if next_extra.startswith('"'):
                    key_part = next_extra[1:]
                    for K in remaining_keys:
                        if (K + '":').startswith(key_part) or key_part.startswith(K + '":'):
                            return True
            return False


class BasePrompt:
    """
    """
    def __init__(
        self,
        func_def_dict: Dict[str, FunctionSchema],
        prompt_path: str = BASE_PROMPT_PATH
        ) -> None:
        self.prompt_path: str = prompt_path
        self.func_def_dict: Dict[str, FunctionSchema] = func_def_dict
        self.base_prompt_str: str = ""

        self._load_base_prompt()
        self._inject_func_def()

    def _load_base_prompt(self) -> None:
        """
        """
        try:
            with open(self.prompt_path, "r", encoding="utf-8") as f:
                self.base_prompt_str = f.read()

        except FileNotFoundError:
            print(f"[ERROR] Base prompt template file not found: {self.prompt_path}")
            sys.exit(1)

    def _inject_func_def(self) -> None:
        """
        """
        funcs_to_inject = [f.model_dump() for name, f in self.func_def_dict.items() if name != "fn_unknown"]
        func_def_text: str = json.dumps(
            funcs_to_inject,
            separators=(',', ':'),
            ensure_ascii=False
        )
        self.base_prompt_str = self.base_prompt_str.format(
            functions_context=func_def_text
        )
    
    def compose_base_prompt(self, user_promp_str: str) -> str:
        """
        """
        return (
            self.base_prompt_str + '"' + user_promp_str + '"\n' + "JSON Output:"
        )

    def get(self) -> str:
        """
        """
        return self.composed_base_prompt


class ModelEngine(Small_LLM_Model):
    """
    """
    def __init__(
            self,
            model_name: str = "Qwen/Qwen3-0.6B",
            **kwargs: Any
            ) -> None:
        super().__init__(model_name=model_name, **kwargs)

        print("Initializing model...\nLooking for tokenizer file... ", end="")
        if os.path.exists(TOKENIZER_PATH):
            print("FOUND!\nLoading data from tokenizer file...")
            with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        
        else:
            print("NOT FOUND\nAccesing to Hugging Face Server... ", end="")
            tokenizer = self.get_path_to_tokenizer_file()
            print("OK")
            with open(tokenizer, "r", encoding="utf-8") as f:
                data = json.load(f)
            print("Downloading tokenizer file...")

            os.makedirs(os.path.dirname(TOKENIZER_PATH), exist_ok=True)
            with open(TOKENIZER_PATH, "w") as f:
                json.dump(data, f, indent=2)
            print("Loading data from tokenizer file...")
        
        self.vocab_dict: Dict[str, int] = data["model"]["vocab"]
        self.id_to_token: Dict[int, str] = {
            v: k for k, v in self.vocab_dict.items()
            }

    def generate(
            self,
            tracker: Optional[TokenizerMaskProtocol],
            prompt: str,
            base_prompt_str: Optional[str] = None,
            max_new_tokens: int = 50,
            answer_str: str = "ANSWER: ",
            printable: bool = False
            ) -> str:
        """
        """
        if base_prompt_str:
            base_prompt_tokes = (
                self.encode(base_prompt_str).flatten().tolist()
            )
            len_base_prompt = len(base_prompt_tokes)
            
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
                        id_to_token=self.id_to_token)

                next_token = int(np.argmax(logits))
                if next_token == EOS_TOKEN_ID: break
                tokens_list.append(next_token)
                current_text = self.decode(tokens_list[len_prompt:])

                if printable:
                    print(f"\r{answer_str}{current_text}", end="", flush=True)

                if tracker and tracker.end_condition(current_text):
                    break

                else: continue

            except Exception as e:
                print(f"ERROR: {e}")
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
        """
        """
        print("\n" + f" GENERATION_TEST_{test_num:02d} ".center(60, "="))
        print("PROMPT: ", test)
        composed_prompt: str = prompt.compose_base_prompt(test)
        if tracker and hasattr(tracker, "setup_prompt"):
            tracker.setup_prompt(test, self)
        generated_text = self.generate(
            tracker=tracker,
            prompt=composed_prompt,
            max_new_tokens=max_new_tokens,
            printable=True)

        return generated_text


def main() -> None:
    """
    """
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument("--functions_definition", default=FUNC_DEF_PATH)
    parser.add_argument("--input", default=FUNC_CALL_TESTS_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)
    arg = parser.parse_args()

    func_def = DynamicFunctionDefinitions(arg.functions_definition)
    user_prompts = UserPrompts(arg.input)
    model = ModelEngine()
    json_traker = JSONStateTracker(func_def)

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
                raise ValueError(f"Unknown or missing function name: {fn_name}")
            
            # Validate parameters using the corresponding Pydantic validator
            func_def.validators[fn_name](**dict_response.get("parameters", {}))
            model_response_list.append(dict_response)
        
        except json.JSONDecodeError as e:
            print(f"JSON Decoding ERROR: {e}")

        except Exception as e:
            print(f"Validation ERROR: {e}")
        
    compose_output_file(model_response_list)


if __name__ == "__main__":
    main()

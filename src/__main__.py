from llm_sdk.llm_sdk import Small_LLM_Model
from typing import Union, Optional, Dict, List, Tuple, Any, Protocol
from pydantic import BaseModel, create_model
import numpy as np
import json
import argparse
import os
import re
from src import __description__


TOKENIZER_PATH = "src/tokenizer.json"                   # attention: manejar esto bien antes de entregar
BASE_PROMPT_PATH = "src/base_prompt.txt"                # attention: terminar de definar correctamente esto antes de entregar
FUNC_DEF_PATH = "data/input/functions_definition.json"
FUNC_CALL_TESTS_PATH = "data/input/function_calling_tests.json"
OUTPUT_PATH = "data/output/function_calls.json"
DEBUG_TOKEN_LIST_PATH = "data/debug/debug_token_list.json"
EOS_TOKEN_ID = 151645


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

    def build_dynamic_param_model(self) -> BaseModel:
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

        param_field: Dict[str, Any] = {}
        for param_name, param_info in self.parameters.items():
            python_type = type_map.get(param_info.type, Any)
            param_field[param_name] = (python_type, ...)
        
        return create_model(f"param_{self.name}", **param_field)


class DynamicFunctionValidator:
    """
    """
    def __init__(self, func_def_dict: Dict[str, FunctionSchema]) -> None:
        self.func_def_dict: Dict[str, FunctionSchema] = func_def_dict

    def validate(self, dict_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        """
        func_name: str = dict_response.get("name")
        if func_name not in self.func_def_dict:
            raise ValueError("Todo error print")
        
        func_schema: FunctionSchema = self.func_def_dict[func_name]
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

        param_field: Dict[str, Any] = {}
        for param_name, param_info in func_schema.parameters.items():
            python_type = type_map.get(param_info.type, Any)
            param_field[param_name] = (python_type, ...)

        ParamDynamicModel: BaseModel = func_schema.build_dynamic_param_model()
        raw_params: Dict[str, Any] = dict_response.get("parameters", {})
        validate_params: BaseModel = ParamDynamicModel(**raw_params)
        dict_response["parameters"] = validate_params

        validated_func_object = FunctionCallResult(**dict_response)
        return validated_func_object.model_dump()


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
    """
    def mask(
            self,
            generated_text: str,
            logits: List[float],
            id_to_token: Dict[int, str]
            ) -> List[float]:
        """
        """
        if len(generated_text) == 0:
            for token_id, token_str in id_to_token.items():
                if token_str != "{":
                    logits[token_id] = -float("inf")

        elif generated_text.endswith("{"):
            for token_id, token_str in id_to_token.items():
                if token_str != '"':
                    logits[token_id] = -float("inf")

        return logits

    @staticmethod
    def end_condition(current_text: str) -> bool:
        """
        """
        try:
            json.loads(current_text)
            return True

        except json.JSONDecodeError: return False


class FileManager:
    """
    """
    def __init__(
            self,
            func_def_path: str = FUNC_DEF_PATH,
            test_prompts_path: str = FUNC_CALL_TESTS_PATH,
            output_path: str = OUTPUT_PATH
            ) -> None:
        self.func_def_path: str = func_def_path
        self.test_prompts_path: str = test_prompts_path
        self.output_path: str = output_path

        self._load_functions_definition()
        self._load_base_prompt()
        self._load_test_prompts()

    def _load_functions_definition(self) -> None:
        """
        """
        try:
            with open(self.func_def_path, "r") as file:
                raw = json.load(file)
                self.func_def: List[FunctionSchema] = [
                    FunctionSchema(**f) for f in raw
                    ]
                self.func_def_dict: Dict[str, FunctionSchema] = {
                    func.name: func for func in self.func_def
                }

        except FileNotFoundError:
            print(f"[Error] Couldn't find {self.func_def_path}")                  # implementar excepciones

        except json.JSONDecodeError:
            print(f"[ERROR] Couldn't parse JSON in {self.func_def_path}")         # implementar excepciones

    def _load_test_prompts(self) -> None:
        """
        """
        try:
            with open(self.test_prompts_path, "r") as file:
                raw: List[Dict[str, str]] = json.load(file)
                self.test_prompts: List[str] = [p["prompt"] for p in raw]

        except FileNotFoundError:
            print(f"[Error] Couldn't find {self.test_prompts_path}")                  # implementar excepciones

        except json.JSONDecodeError:
            print(f"[ERROR] Couldn't parse JSON in {self.test_prompts_path}")         # implementar excepciones

    def _load_base_prompt(self) -> None:
        """
        """
        try:
            with open(BASE_PROMPT_PATH, "r", encoding="utf-8") as f:
                self.base_prompt_str: str = f.read()

        except FileNotFoundError:
            print(f"[Error] Couldn't find {BASE_PROMPT_PATH}")        # implementar excepciones

        except json.JSONDecodeError:
            print(f"[ERROR] Couldn't parse JSON in {BASE_PROMPT_PATH}")        # implementar excepciones

    def compose_output_file(self, model_response_list: List[Dict[str, Any]]) -> None:
        """
        """
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(
                model_response_list,
                fp=f,
                indent=2,
                ensure_ascii=False
                )

    @staticmethod
    def debug_output_token_list(debug_token_list: List[str]) -> None:
        """
        """
        os.makedirs(os.path.dirname(DEBUG_TOKEN_LIST_PATH), exist_ok=True)
        with open(DEBUG_TOKEN_LIST_PATH, "w", encoding="utf-8") as f:
            json.dump(
                debug_token_list,
                fp=f,
                indent=2,
                ensure_ascii=False
                )


class BasePrompt:
    """
    """
    def __init__(
            self,
            base_prompt_str: str,
            func_def: List[FunctionSchema],
            user_promp: str
            ) -> None:
        self.func_def_text: str = json.dumps(
            [f.model_dump() for f in func_def],
            separators=(',', ':'),
            ensure_ascii=False
        )
        base_prompt_str = base_prompt_str.format(
        functions_context=self.func_def_text
        )
        self.composed_base_prompt = (
            base_prompt_str + '"' + user_promp + '"\n' + "JSON Output:"
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
            **kwargs: Any) -> None:
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
        # self.debug_chosen_tokens: List[str] = []

    def generate(
            self,
            tracker: Optional[TokenizerMaskProtocol],
            prompt: BasePrompt,
            max_new_tokens: int = 50,
            answer_str: str = "ANSWER: ",
            printable: bool = False) -> str:
        """
        """        
        prompt_tokens = self.encode(prompt.get()).flatten().tolist()
        len_prompt = len(prompt_tokens)
        tokens_list = prompt_tokens.copy()

        for _ in range(max_new_tokens):
            
            generated_tokens = tokens_list[len_prompt:]
            generated_text = self.decode(generated_tokens)
            logits = self.get_logits_from_input_ids(tokens_list)

            if tracker:
                logits = tracker.mask(
                    generated_text=generated_text,
                    logits=logits,
                    id_to_token=self.id_to_token)

            next_token = int(np.argmax(logits))
            if next_token == EOS_TOKEN_ID: break
            # self.debug_chosen_tokens.append(self.decode(next_token))
            tokens_list.append(next_token)
            current_text = self.decode(tokens_list[len_prompt:])

            if printable:
                print(f"\r{answer_str}{current_text}", end="", flush=True)

            if tracker and tracker.end_condition(current_text):
                break

            else: continue

        return self.decode(tokens_list[len_prompt:])
    
    def test_model(
            self,
            tracker: Optional[TokenizerMaskProtocol],
            prompt: BasePrompt,
            test: str,
            test_num: int = 0) -> str:
        """
        """
        print("\n" + f" GENERATION_TEST_{test_num:02d} ".center(60, "="))
        print("PROMPT: ", test)
        generated_text = self.generate(
            tracker=tracker,
            prompt=prompt,
            max_new_tokens=100,
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
    
    file = FileManager(
        func_def_path=arg.functions_definition,
        test_prompts_path=arg.input,
        output_path=arg.output
    )
    model = ModelEngine()
    json_traker = JSONStateTracker()
    validator = DynamicFunctionValidator(file.func_def_dict)
    model_response_list: List[Dict[str, Any]] = []

    for i, test in enumerate(file.test_prompts):
        prompt = BasePrompt(
            base_prompt_str=file.base_prompt_str,
            func_def=file.func_def,
            user_promp=file.test_prompts[i]
        )
        generated_text = model.test_model(
            prompt=prompt,
            tracker=json_traker,
            test=test,
            test_num=i
            )
        dict_response = json.loads(generated_text)
        
        try:
            validated_result = validator.validate(dict_response)
            model_response_list.append(validated_result)
        
        except json.JSONDecodeError as e:
            print(f"JSON Decoding ERROR: {e}")

        except Exception as e:
            print(f"Validation ERROR: {e}")
        
    file.compose_output_file(model_response_list)
    # file.debug_output_token_list(model.debug_chosen_tokens)


if __name__ == "__main__":
    main()

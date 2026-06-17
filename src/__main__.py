from llm_sdk.llm_sdk import Small_LLM_Model
from typing import Union, Optional, Dict, List, Tuple, Any
from pydantic import BaseModel
import numpy as np
import json
import argparse
import os
import re
from src import __description__


TOKENIZER_PATH = "src/tokenizer.json"                   # attention: manejar esto bien antes de entregar
BASE_PROMPT_PATH = "src/BASE_PROMPT.txt"                # attention: terminar de definar correctamente esto antes de entregar
EOS_TOKEN_ID = 151645


class ParameterInfo(BaseModel):
    """
    """
    type: str


class FunctionSchema(BaseModel):
    """
    """
    name: str
    description: str
    parameters: Dict[str, ParameterInfo]
    returns: Dict[str, str]


class FunctionCallResult(BaseModel):
    """
    """
    prompt: str
    name: str
    parameters: Dict[str, Any]


class JSONStateTracker:
    """
    """
    def __init__(
            self,
            user_prompt: str,
            func_def_dict: List[FunctionSchema]
            ) -> None:
        self.user_promp: str = user_prompt
        self.func_def_dic: Dict[str, FunctionSchema] = func_def_dict

    def mask(
            self,
            generated_text: str,
            logits: int,
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


class ModelEngine(Small_LLM_Model):
    """
    """
    def __init__(
            self,
            func_def_path: str,
            test_prompts_path: str,
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

        self._load_functions_definition(func_def_path)
        self._load_test_prompts(test_prompts_path)
        self._load_base_prompt()

    def _load_functions_definition(self, path: str) -> None:
        """
        """
        try:
            with open(path, "r") as file:
                raw = json.load(file)
                self.func_def: List[FunctionSchema] = [
                    FunctionSchema(**f) for f in raw
                    ]
                self.func_def_text: str = json.dumps(
                    [f.model_dump() for f in self.func_def],
                    separators=(',', ':'),
                    ensure_ascii=False
                )
                self.func_def_dict: Dict[str, FunctionSchema] = {
                    func.name: func for func in self.func_def
                }

        except FileNotFoundError:
            print(f"[Error] Couldn't find {path}")                  # implementar excepciones

        except json.JSONDecodeError:
            print(f"[ERROR] Couldn't parse JSON in {path}")         # implementar excepciones

    def _load_test_prompts(self, path: str) -> None:
        """
        """
        try:
            with open(path, "r") as file:
                raw: List[Dict[str, str]] = json.load(file)
                self.test_prompts: List[str] = [p["prompt"] for p in raw]

        except FileNotFoundError:
            print(f"[Error] Couldn't find {path}")                  # implementar excepciones

        except json.JSONDecodeError:
            print(f"[ERROR] Couldn't parse JSON in {path}")         # implementar excepciones

    def _load_base_prompt(self) -> None:
        """
        """
        try:
            with open(BASE_PROMPT_PATH, "r", encoding="utf-8") as f:
                self.base_prompt: str = f.read()
                self.base_prompt = self.base_prompt.format(
                    functions_context=self.func_def_text
                    )

        except FileNotFoundError:
            print(f"[Error] Couldn't find {BASE_PROMPT_PATH}")        # implementar excepciones

        except json.JSONDecodeError:
            print(f"[ERROR] Couldn't parse JSON in {BASE_PROMPT_PATH}")        # implementar excepciones

    def _compose_base_prompt_with_user_prompt(self, user_promp: str) -> str:
        """
        """
        return (
            self.base_prompt + '"' + user_promp + '"\n' + "JSON Output:"
        )

    def generate(
            self,
            max_new_tokens: int = 50,
            user_prompt: str = Optional[str]) -> str:
        """
        """
        if user_prompt is None:
            user_prompt = self.test_prompts[0]

        prompt = self._compose_base_prompt_with_user_prompt(user_prompt)
        
        prompt_tokens = self.encode(prompt).flatten().tolist()
        len_prompt = len(prompt_tokens)
        tokens_list = prompt_tokens.copy()

        traker = JSONStateTracker(
            user_prompt=user_prompt,
            func_def_dict=self.func_def_dict
        )

        for _ in range(max_new_tokens):
            
            generated_tokens = tokens_list[len_prompt:]
            generated_text = self.decode(generated_tokens)

            logits = self.get_logits_from_input_ids(tokens_list)
            logits = traker.mask(
                generated_text=generated_text,
                logits=logits,
                id_to_token=self.id_to_token)

            next_token = int(np.argmax(logits))
            
            if next_token == EOS_TOKEN_ID:
                break

            tokens_list.append(next_token)
            current_text = self.decode(tokens_list[len_prompt:])

            try:
                json.loads(current_text)
                break
            except json.JSONDecodeError:
                continue

        return self.decode(tokens_list[len_prompt:])


def main() -> None:
    """
    """
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument(
        "--functions_definition", default="data/input/functions_definition.json"
        )
    parser.add_argument(
        "--input", default="data/input/function_calling_tests.json"
        )
    parser.add_argument(
        "--output", default="data/output/function_calls.json"
        )
    arg = parser.parse_args()
    

    model = ModelEngine(
        func_def_path=arg.functions_definition,
        test_prompts_path=arg.input,
    )
    for i, test in enumerate(model.test_prompts):
        test_model(model, test, test_num=i)


def test_model(
        model: ModelEngine,
        test: str,
        test_num: int = 0) -> None:
    """
    """
    generated_text = model.generate(user_prompt=test)
    print("\n" + f" GENERATION_TEST_{test_num:02d} ".center(60, "="))
    print("PROMPT: ", test)
    print("ANSWER: ", generated_text)



if __name__ == "__main__":
    main()

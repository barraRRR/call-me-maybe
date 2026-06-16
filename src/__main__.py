from llm_sdk.llm_sdk import Small_LLM_Model
from typing import Union, Optional, Dict, List, Tuple, Any
from pydantic import BaseModel
import numpy as np
import json
import argparse
import os
from src import __description__


TOKENIZER_PATH = "src/tokenizer.json"                   # attention: manejar esto bien antes de entregar
BASE_PROMPT_PATH = "src/BASE_PROMPT.txt"                # attention: terminar de definar correctamente esto antes de entregar


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
            print("Loading data from tokenizer file...")

            os.makedirs(os.path.dirname(TOKENIZER_PATH), exist_ok=True)
            with open(TOKENIZER_PATH, "w") as f:
                json.dump(data, f, indent=2)
        
        vocab_dict: Dict[str, int] = data["model"]["vocab"]
        self.id_to_token: Dict[int, str] = {v: k for k, v in vocab_dict.items()}

        self._load_functions_definition(func_def_path)
        self._load_test_prompts(test_prompts_path)
        self._load_base_prompt()

    def _load_functions_definition(self, path: str) -> None:
        """
        """
        try:
            with open(path, "r") as file:
                # Desempaquetamos el diccionario para instanciar el modelo de Pydantic
                raw = json.load(file)
                self.func_def: List[FunctionSchema] = [FunctionSchema(**f) for f in raw]
                self.func_def_text: str = json.dumps(
                    [f.model_dump() for f in self.func_def],
                    indent=2,
                    ensure_ascii=False
                )

        except FileNotFoundError:
            print(f"[Error] Couldn't find {path}")        # implementar excepciones

        except json.JSONDecodeError:
            print(f"[ERROR] Couldn't parse JSON in {path}")        # implementar excepciones

    def _load_test_prompts(self, path: str) -> None:
        """
        """
        try:
            with open(path, "r") as file:
                raw: List[Dict[str, str]] = json.load(file)
                self.test_prompts: List[str] = [p["prompt"] for p in raw]

        except FileNotFoundError:
            print(f"[Error] Couldn't find {path}")        # implementar excepciones

        except json.JSONDecodeError:
            print(f"[ERROR] Couldn't parse JSON in {path}")        # implementar excepciones

    def _load_base_prompt(self) -> None:
        """
        """
        try:
            with open(BASE_PROMPT_PATH, "r", encoding="utf-8") as f:
                self.base_prompt: str = f.read()
            
            self.base_prompt = self.base_prompt.format(
            functions_context=self.func_def_text,
            user_query=self.test_prompts[0]             # TEMP
            )

        except FileNotFoundError:
            print(f"[Error] Couldn't find {BASE_PROMPT_PATH}")        # implementar excepciones

        except json.JSONDecodeError:
            print(f"[ERROR] Couldn't parse JSON in {BASE_PROMPT_PATH}")        # implementar excepciones

    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        """
        """
        tokens = self.encode(prompt)
        
        # Al saber que es un torch.Tensor, podemos aplanarlo a 1D y convertirlo a lista nativa
        tokens_list = tokens.flatten().tolist()

        # Cambiamos a <|im_end|> que es el finalizador habitual en modo Chat para Qwen
        EOS_TOKEN_ID = 151645

        for _ in range(max_new_tokens):
            
            logits = self.get_logits_from_input_ids(tokens_list)
            
            # C. Buscamos el índice (el ID del token) con la puntuación más alta
            next_token = int(np.argmax(logits))
            
            # D. Comprobamos si el modelo ha decidido terminar de hablar
            if next_token == EOS_TOKEN_ID:
                break
                
            # E. Si no ha terminado, añadimos el nuevo token a nuestra lista
            # para que en la siguiente vuelta el modelo tenga más contexto
            tokens_list.append(next_token)
            
        # 3. Fuera del bucle, decodificamos la lista completa de vuelta a texto
        generated_text = self.decode(tokens_list)
        return generated_text


def main() -> None:
    """
    """
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument("--functions_definition", default="data/input/functions_definition.json")
    parser.add_argument("--input", default="data/input/function_calling_tests.json")
    parser.add_argument("--output", default="data/output/function_calls.json")
    arg = parser.parse_args()
    

    model = ModelEngine(
        func_def_path=arg.functions_definition,
        test_prompts_path=arg.input,
    )
    """
    print("\n" + " MODEL TEST PROMPTS ".center(60, "="))
    print(model.test_prompts)

    print("\n" + " FUNC DEFINITIONS ".center(60, "="))
    print(model.func_def)

    print("\n" + " FORMATED PROMPT ".center(60, "="))
    print(model.base_prompt)
    """
    generated_text = model.generate(model.base_prompt)

    print("\n" + " GENERATION TEST ".center(60, "="))
    print(generated_text)


if __name__ == "__main__":
    main()

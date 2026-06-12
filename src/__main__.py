from llm_sdk.llm_sdk import Small_LLM_Model
from typing import Union, Optional, Dict, List, Tuple, Any
from pydantic import BaseModel
import numpy as np
import json
import argparse
import os


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
    TEST_TOKENIZER_PATH = "debug/test_tokenizer.json"

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B", **kwargs) -> None:
        super().__init__(model_name=model_name, **kwargs)

        tokenizer = self.get_path_to_tokenizer_file()
        with open(tokenizer, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        os.makedirs(os.path.dirname(self.TEST_TOKENIZER_PATH), exist_ok=True)
        with open(self.TEST_TOKENIZER_PATH, "w") as f:
            json.dump(data, f, indent=2)

    def load_functions_definition(self, path: str) -> None:
        """
        """
        try:
            with open(path, "r") as file:
                # Desempaquetamos el diccionario para instanciar el modelo de Pydantic
                raw = json.load(file)
                self.func_def: List[FunctionSchema] = [FunctionSchema(**f) for f in raw]

        except FileNotFoundError:
            print(f"[Error] Couldn't find {path}")        # implementar excepciones

        except json.JSONDecodeError:
            print(f"[ERROR] Couldn't parse JSON in {path}")        # implementar excepciones

    def load_test_prompts(self, path: str) -> None:
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

    def load_base_prompt(self, path: str = "BASE_PROMPT.txt") -> None:
        """
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw: str = json.load(f)
                self.base_prompt: str = raw

        except FileNotFoundError:
            print(f"[Error] Couldn't find {path}")        # implementar excepciones

        except json.JSONDecodeError:
            print(f"[ERROR] Couldn't parse JSON in {path}")        # implementar excepciones

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
            
            # Identificamos la forma del tensor/array para extraer correctamente la probabilidad del vocabulario
            if hasattr(logits, "ndim") and logits.ndim == 3:
                last_logits = logits[0, -1, :]  # [batch, seq, vocab] -> Extrae el último token
            elif hasattr(logits, "ndim") and logits.ndim == 2:
                last_logits = logits[-1, :]     # [seq, vocab] -> Extrae el último token
            else:
                last_logits = logits            # [vocab] -> Asumimos que ya es el array 1D
                
            # Convertimos de PyTorch a NumPy de forma segura (si fuera necesario) antes del argmax
            if hasattr(last_logits, "detach"):
                last_logits = last_logits.detach().cpu().numpy()
            
            # C. Buscamos el índice (el ID del token) con la puntuación más alta
            next_token = int(np.argmax(last_logits))
            
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
    parser = argparse.ArgumentParser(description="A function calling tool that translates natural language prompts into structured function calls.")
    parser.add_argument("--functions_definition", default="data/input/functions_definition.json")
    parser.add_argument("--input", default="data/input/function_calling_tests.json")
    parser.add_argument("--output", default="data/output/function_calls.json")
    arg = parser.parse_args()
    

    model = ModelEngine()
    model.load_functions_definition(arg.functions_definition)
    model.load_test_prompts(arg.input)

    generated_text = model.generate(model.base_promt, 200)

    print("\n" + " TEST 02 ".center(60, "="))
    print(f"ENCODING      : {TEST_02}")
    print(f"RESPONSE      :", generated_text)


if __name__ == "__main__":
    main()

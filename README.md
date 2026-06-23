*This project has been created as part of the 42 curriculum by jbarreir.*

# call me maybe

<div align="center">

<h1>☎️ here's my prompt, so call me maybe</h1>

#### *Guided Function Calling for Small LLMs via Constrained Decoding*

![Version](https://img.shields.io/badge/version-v1.0.0-blue.svg)
![Python Versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

</div>

## Description

**call me maybe** is an introduction to function calling in Large Language Models (LLMs) using constrained decoding techniques. Small language models (such as the 500 million parameter `Qwen/Qwen3-0.6B` used in this project) are notoriously unreliable at producing structured JSON outputs spontaneously. When prompted to produce JSON, they might succeed only 30% of the time. Yet production systems achieve 99%+ reliability with these same small models.

How? The answer lies in **constrained decoding**—a technique that guides the model's output token-by-token to guarantee valid structure. This project implements a custom guided state-machine decoder that dynamically restricts the LLM's output logits during generation, guaranteeing 100% syntactically valid JSON outputs that comply perfectly with dynamic function schemas.

---

## Instructions

### Prerequisites

- Python 3.10+
- `uv` (Fast Python package installer)

### Installation

Use the provided `Makefile` to automatically set up the virtual environment, install the required dependencies (`pydantic`, `numpy`, etc.), and fetch the necessary data archives:
```bash
make install
```

### Execution

Run the default generation pipeline (which processes the public test inputs):
```bash
make run
```

You can also run custom inputs by supplying arguments directly to the module:
```bash
uv run python -m src \
  --functions_definition <path_to_definitions_json> \
  --input <path_to_tests_json> \
  --output <path_to_output_json>
```

To clean up caches and generated files:
```bash
make clean
```

---

## Algorithm Choices & Guided Decoding Strategy

The core engine relies on a custom [ConstrainedJSONTracker](file:///Users/barreiro/coding/42/call_me_maybe/src/__main__.py#L251) to restrict output token probabilities at each step.

### 1. Phase-Based Logits Restriction
The generation pipeline forces the output through a series of deterministic steps to guarantee JSON validity:
- **Prefix Injection:** We initialize the generation prefix to `{"prompt": "<user_prompt>", "name": "`. 
- **Name Phase:** During this phase, the tracker limits the allowed tokens to those that can build valid function names matching the catalog in [DynamicFunctionDefinitions](file:///Users/barreiro/coding/42/call_me_maybe/src/__main__.py#L55).
- **Suffix Injection:** As soon as a function name is completed, we inject the parameter header suffix: `","parameters":{`.

### 2. State-Machine Driven Parameter Parsing (`JSONState`)
Once in the parameter dictionary context, the tracker operates as a state machine governed by the current generated character sequence. In [_scan_parameters](file:///Users/barreiro/coding/42/call_me_maybe/src/__main__.py#L515), we scan the parameters text to determine the current `JSONState`:

- **`KEY_START` & `KEY_PARTIAL`:** Restricts vocabulary candidate tokens to parameter names belonging to the schema of the active function (that have not been generated yet).
- **`COLON_START`:** Restricts next tokens strictly to `:`.
- **`VALUE_START` & `VALUE_PARTIAL`:** Constrains candidate tokens based on the parameter's expected type:
  - *String Type:* Allows string tokens starting with quotes. For regular expressions (`regex`), it prevents wildcards (such as `.*`) and leading spaces from matching initially.
  - *Number Type:* Matches candidate tokens against a floating-point/integer pattern ([number_regex](file:///Users/barreiro/coding/42/call_me_maybe/src/utils.py#L17)).
  - *Boolean Type:* Dynamically filters vocabulary candidates to only allow completions spelling `true` or `false`.
- **`COMMA_OR_CLOSE_START`:** Allows a `,` if there are remaining parameters in the schema, or a closing brace `}`.
- **`CLOSED`:** Closes the final prompt schema and terminates the generation by enforcing the End-Of-String token (`EOS_TOKEN_ID`).

> **💡 Time Complexity Note:** By keeping all valid tokens cached in memory and checking prefix strings, the logits masking function [mask](file:///Users/barreiro/coding/42/call_me_maybe/src/__main__.py#L280) runs in **$O(V)$** time per step (where $V$ is the vocabulary size, approximately 151,000 tokens for Qwen). This ensures minimal overhead and near-instant token generation.

---

## Dynamic Schema Parsing & Data Validation

Instead of hardcoding schemas, this project uses **Pydantic** models to load, parse, and validate function signatures dynamically at runtime.

### Dynamic Validation Generation
When [DynamicFunctionDefinitions](file:///Users/barreiro/coding/42/call_me_maybe/src/__main__.py#L55) is initialized:
1. It ingests definitions from `functions_definition.json`.
2. It dynamically compiles validation models for each loaded function using Pydantic's `create_model` function in [_preconfigure_validators](file:///Users/barreiro/coding/42/call_me_maybe/src/__main__.py#L107):
   ```python
   ParamDynamicModel = create_model(f"params_{func_name}", **param_fields)
   self.validators[func_name] = ParamDynamicModel
   ```
3. Once the LLM generates a function call, the argument dictionary is validated against the dynamic model. If valid, Pydantic's `model_dump()` coerces and formats parameter types (e.g., converting integers to floats where `float` type is expected) before saving.

### Structural vs. Semantic Defense
We removed all fallback references to `fn_unknown`. Because the guided tracker enforces 100% correct JSON formats and schema compliance, the LLM is prevented from generating invalid function names or malformed arguments. Removing `fn_unknown` simplifies the model's choices, leading to more accurate function selection and parameter extraction.

---

## Performance Benchmarks

Below is a performance overview of the 11 tests from the public exercise set:

| Test # | Prompt Description | Function Selected | Parameter Outputs | Status |
|--------|--------------------|-------------------|-------------------|--------|
| **1** | What is the sum of 2 and 3? | `fn_add_numbers` | `{"a": 2.0, "b": 3.0}` | **VALID (100%)** |
| **2** | What is the sum of 265 and 345? | `fn_add_numbers` | `{"a": 265.0, "b": 345.0}` | **VALID (100%)** |
| **3** | Greet shrek | `fn_greet` | `{"name": "shrek"}` | **VALID (100%)** |
| **4** | Greet john | `fn_greet` | `{"name": "john"}` | **VALID (100%)** |
| **5** | Reverse the string 'hello' | `fn_reverse_string` | `{"s": "hello"}` | **VALID (100%)** |
| **6** | Reverse the string 'world' | `fn_reverse_string` | `{"s": "world"}` | **VALID (100%)** |
| **7** | What is the square root of 16? | `fn_get_square_root` | `{"a": 16.0}` | **VALID (100%)** |
| **8** | Calculate the square root of 144 | `fn_get_square_root` | `{"a": 144.0}` | **VALID (100%)** |
| **9** | Replace numbers with "NUMBERS" | `fn_substitute_string_with_regex` | `{"source_string": "...", "regex": "\\d+", "replacement": "NUMBERS"}` | **VALID (100%)** |
| **10** | Replace vowels with asterisks | `fn_substitute_string_with_regex` | `{"source_string": "...", "regex": "[aeiouAEIOU]", "replacement": "*"}` | **VALID (100%)** |
| **11** | Substitute 'cat' with 'dog' | `fn_substitute_string_with_regex` | `{"source_string": "...", "regex": "\\bcat\\b", "replacement": "dog"}` | **VALID (100%)** |

*All 11 public tests pass with 100% correct type validation and argument coercion.*

---

## Challenges Faced

- **String Alignment under Token Escapes:** String sequences are represented differently in raw text versus decoded tokens (e.g. carriage returns or quotation marks). Using raw text lengths for character-based slicing resulted in boundary errors. We resolved this by computing a `decoded_prefix` tracking the exact decoded representations.
- **Greedy Regular Expression Patterns:** The small model tends to generate wildcards (`.*`) at the beginning of regular expression arguments. We successfully prevented this by dynamically blocking tokens starting with `.` or `*` at the beginning of value strings when the target parameter is named `regex`.

---

## Resources & References

### Documentation
- **Pydantic**: [docs.pydantic.dev](https://docs.pydantic.dev) - Ingesting dynamic JSON structures.
- **LLM Logits Restriction**: Guided decoding and logit masking patterns for autoregressive LLMs.

### AI Usage Disclosure
Artificial Intelligence was utilized in the following capacities:
- **State Boundaries Tuning:** AI assisted in analyzing token alignments (resolving escape sequences and carriage return bugs).
- **Dynamic Pydantic Compilations:** Assisted in structuring the `create_model` invocations inside the dynamic validator loader.
- **Documentation:** AI assisted in formatting this README to mirror the template of our previous project and highlight core algorithms.

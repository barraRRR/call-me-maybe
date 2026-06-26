<div align="center">


# call me maybe

<img width="684" height="387" alt="call-me-baybe_kv" src="https://github.com/user-attachments/assets/f70990ca-d7a9-44bb-84d4-90abf4190b6f" />
  
### *☎️ here's my prompt, so call me maybe*
![Version](https://img.shields.io/badge/version-v1.0.0-blue.svg)
![Python Versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![42 School](https://img.shields.io/badge/school-42-black?logo=42&logoColor=white)
![Success](https://img.shields.io/badge/success-111%2F100-green.svg)
![Qwen](https://img.shields.io/badge/LLM-Qwen-orange?logo=alibabacloud&logoColor=white)


</div>

## Description

**call me maybe** is an introduction to function calling in Large Language Models (LLMs) using constrained decoding techniques. Small language models (such as the 500 million parameter `Qwen/Qwen3-0.6B` used in this project) are notoriously unreliable at producing structured JSON outputs spontaneously. When prompted to produce JSON, they might succeed only 30% of the time. Yet production systems achieve 99%+ reliability with these same small models.

How? The answer lies in **constrained decoding**—a technique that guides the model's output token-by-token to guarantee valid structure. This project implements a custom guided state-machine decoder that dynamically restricts the LLM's output logits during generation, guaranteeing 100% syntactically valid JSON outputs that comply perfectly with dynamic function schemas.

### Contents
- [Instructions](#instructions)
- [Algorithm Explanation](#algorithm-explanation)
- [Design Decisions](#design-decisions)
- [Dynamic Schema Parsing & Data Validation](#dynamic-schema-parsing--data-validation)
- [Performance Analysis](#performance-analysis)
- [Challenges Faced](#challenges-faced)
- [Testing Strategy](#testing-strategy)
- [Example Usage](#example-usage)
- [Resources & References](#resources--references)

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

## Algorithm Explanation

The core problem this project solves is: **how do you force a 0.6B-parameter model, which is fundamentally a probability distribution over the next token, to only ever emit syntactically valid, schema-compliant JSON — without retraining it, without retrying failed generations, and without post-hoc fixing?**

The answer is **constrained decoding via logits masking**. At every single generation step, before the model picks its next token via `argmax`, we intercept the raw logits vector and set every token that *would* break the JSON grammar or violate the active function's schema to `-inf`. This guarantees the sampled token (always the highest-probability *legal* token) can never produce malformed output — not 99.9% of the time, but 100% of the time, by construction. The model's only real "decision" is semantic: which function to pick and what values to fill in. The decoder's grammar is never a place where it can go wrong.

This is implemented entirely inside `ConstrainedJSONTracker`, which the generation loop in `ModelEngine.generate` calls on every step:

```python
logits = self.get_logits_from_input_ids(tokens_list)
if tracker:
    logits = tracker.mask(generated_text, logits, id_to_token)
next_token = int(np.argmax(logits))
```

The tracker doesn't touch model weights or attention — it only ever reasons about **text**: the string generated so far, decoded fresh at every step, and the static catalog of function schemas. This text-first design is deliberate (see [Design Decisions](#design-decisions)) and is what keeps the whole system decoupled from tokenizer internals almost everywhere except the prefix-matching step.

### 1. Phase-Based Generation

Generation is split into four ordered phases, each with a different masking strategy. The transition between phases is driven purely by how much of the expected text has already been generated — there is no explicit phase counter; the tracker re-derives "where we are" from the decoded string on every call.

**Phase 1 — Prefix Injection.** Before generation even starts, [setup_prompt](file:///Users/barreiro/coding/42/call_me_maybe/src/__main__.py#L361) pre-computes the fixed opening of the JSON object:

```python
self.prefix = '{"prompt":' + json.dumps(user_prompt) + ',"name":"'
```

Since `user_prompt` is already known before the model generates a single token, there is nothing to predict here — `json.dumps` handles all escaping (quotes, backslashes, unicode) automatically, so the prompt is injected byte-for-byte correct. While `len(generated_text) < len(decoded_prefix)`, `_build_rem_prefix` scans the vocabulary for tokens whose decoded string is a prefix of (or is prefixed by) the remaining slice of `decoded_prefix`, steering the model to reproduce the deterministic header one token at a time without it ever needing to "choose" anything.

**Phase 2 — Name Phase.** Once the prefix is fully emitted, `_define_name_phase` checks the generated suffix against every `fn_name + param_suffix` string in the catalog (where `param_suffix = '","parameters":{'`). If the suffix is a strict prefix of any candidate, we're still mid-name; `_resolve_name_phase` then restricts the vocabulary to tokens that keep at least one candidate function name alive. Critically, this means **the model cannot spell a function name that doesn't exist** — there is no `fn_unknown` fallback because the grammar itself makes hallucinated names structurally impossible to generate.

**Phase 3 — Suffix Injection.** The instant the generated suffix exactly matches `fn_name + '","parameters":{'` for some function, that function becomes the `active_fn` and the tracker moves into parameter parsing for that specific schema.

**Phase 4 — Parameter State Machine.** This is where most of the complexity lives (detailed below).

### 2. The Parameter State Machine (`JSONState`)

Once inside the parameters object for the active function, `_scan_parameters` re-parses the parameter text generated so far, character by character, on **every single step**, to recover the current `JSONState`:

| State | Meaning | What gets masked next |
|---|---|---|
| `KEY_START` / `KEY_PARTIAL` | Expecting a new key, or mid-way through one | Only tokens that can complete one of the **remaining, not-yet-emitted** parameter names from the schema |
| `COLON_START` | A key just closed | Only the `:` character |
| `VALUE_START` / `VALUE_PARTIAL` | Expecting / mid-way through a value | Only tokens consistent with that parameter's declared type |
| `COMMA_OR_CLOSE_START` | A value just completed | A comma if parameters remain, or the closing brace if none do |
| `CLOSED` | All parameters emitted, closing brace consumed | The literal closing brace for the parameters dict, then the model's `EOS_TOKEN_ID` to stop generation entirely |

Re-scanning from scratch each step (rather than keeping a running state object) trades a small amount of redundant computation for robustness: the state is always re-derived from ground truth (the actual decoded string so far), so there's no possibility of the tracker's internal state silently drifting from what the model has actually emitted.

**Type-aware value masking** is the part that does the real schema-enforcement work, in `_get_allowed_token_ids` and its helpers `_is_valid_value_prefix` / `_is_valid_separator_prefix`:
- **String values** — every vocabulary token is checked via `find_first_unescaped_quote` (which walks backwards counting backslashes via `is_quote_escaped`. A token is accepted if it either continues the string body or closes it with an unescaped quote followed by a valid separator.
- **Number values** — candidate tokens are matched against `number_regex` concatenated with what's been generated so far. A token is also accepted if the value generated *so far* is already a syntactically complete number (checked by `is_complete_number`, which rejects dangling states like a lone sign or decimal point) and the token itself looks like a valid separator.
- **Boolean values** — the candidate set is dynamically narrowed to only the tokens that keep `true` or `false` (whichever the first emitted character commits to) spellable to completion.
- **Regex guard** — for any parameter literally named `regex`, the very first character of the value is blocked from being a wildcard-style character, which structurally prevents the classic small-model failure mode of opening every pattern with a greedy wildcard (see [Challenges Faced](#challenges-faced)).

### 3. Termination

`end_condition` performs a cheap JSON-parse attempt on the generated text after every accepted token; the moment the full structure parses as valid JSON, generation halts even if `max_new_tokens` hasn't been reached. Combined with the `CLOSED` state forcing `EOS_TOKEN_ID` once the final closing brace is emitted, this gives a double-layered, redundant stop condition — belt and suspenders against any edge case where one of the two signals might fire too late.

> **💡 Complexity note:** Because `mask` iterates over the full vocabulary (~151,000 entries for Qwen3-0.6B) at every step and does only constant-time string prefix checks per token (no regex backtracking, no recursive parsing), the per-step cost of masking is **O(V)**, where V is vocabulary size. This is independent of how deep into the JSON structure generation currently is, so masking overhead stays flat for the whole generation — it doesn't get slower as the output grows.

---

## Design Decisions

A few choices shaped this implementation more than any others. Documenting them here both explains *why* the code looks the way it does and gives a sense of the trade-offs considered.

**Text-based state recovery over token-based state tracking.** The tracker decodes the generated tokens to a string and re-parses that string from scratch on every step, rather than maintaining incremental state (e.g. "we are 3 keys into the dict, currently inside a string value"). This is slightly more expensive per step, but it means the source of truth is always the literal text the model produced — not an assumption about what *should* have been produced. Any discrepancy between the tokenizer's byte-pair-encoding boundaries and the logical JSON grammar boundaries (which is exactly the kind of bug this class of system is prone to) self-corrects on the next step instead of compounding.

**Schema-driven validators generated at runtime, not hand-written.** ´DynamicFunctionDefinitions._preconfigure_validators´ uses Pydantic's `create_model` to synthesize one validation model per function directly from `functions_definition.json`:

```python
ParamDynamicModel = create_model(f"params_{func_name}", **param_fields)
```

This means adding a new function to the catalog requires editing exactly one file — the JSON definitions — and nothing in `__main__.py` needs to change. The decoding grammar, the prompt's function catalog, and the post-generation validation all derive from that single source of truth.

**Two layers of defense instead of one.** The constrained decoder guarantees *structural* validity (correct JSON syntax, correct types per the grammar). It does **not** guarantee *semantic* correctness (e.g. that `"a": 2.0` is actually the right value for "the sum of 2 and 3"). That second layer is handled after generation, in `main()`, by re-validating the parsed dictionary against the same Pydantic model and coercing types (`model_dump()`) before writing output. Decoding-time constraints and post-hoc validation check different things, so both stages stay in the pipeline rather than assuming the first makes the second redundant.

**Greedy decoding (`argmax`) rather than sampling.** Since the goal is reliability and reproducibility on a constrained task, every step picks the single highest-logit legal token rather than sampling from a temperature-scaled distribution. This trades output diversity (irrelevant here, since each test prompt has one canonical correct call) for fully deterministic, repeatable runs — useful for grading and for the benchmark table below.

**Centralized error templates.** All user-facing error strings live in `error_handling.json` and are loaded once via ´load_error_messages´, keyed by a short identifier and formatted with `.format()` placeholders. This keeps error wording out of the control-flow code and makes the failure modes of the pipeline (missing files, malformed JSON, schema validation errors, unknown function names) auditable in one place rather than scattered across `try/except` blocks.

---

## Dynamic Schema Parsing & Data Validation

Schemas are never hardcoded. ´DynamicFunctionDefinitions´ loads `functions_definition.json`, parses each entry into a ´FunctionSchema´ Pydantic model, and compiles one runtime validator per function (see [Design Decisions](#design-decisions) for why). That single JSON file is the one place that needs editing to add, remove, or modify a function — the prompt's injected function catalog, the decoding grammar, and the post-generation validators all read from the same `func_def_dict`.

---

## Performance Analysis

### Accuracy
All 11 public tests resolve to a syntactically valid, schema-compliant JSON object, and in every case the model selects the function the prompt actually intends and fills in arguments matching the expected values. Since the constrained decoder makes invalid JSON structurally unreachable, "accuracy" here has two independent components worth separating:

- **Structural validity: 100%, guaranteed by construction.** This isn't a measured rate that could regress — it's an invariant of the masking algorithm. As long as the schema and grammar logic are correct, no test input can produce malformed JSON, because every token that would do so has logit `-inf` at generation time.
- **Semantic correctness (right function, right arguments): 11/11 on the public set.** This is the part that actually depends on the underlying model's language understanding, and it's what the benchmark table below tracks.

| Test # | Prompt | Function Selected | Parameter Outputs | Status |
|--------|--------------------|-------------------|-------------------|--------|
| 1 | What is the sum of 2 and 3? | `fn_add_numbers` | `{"a": 2.0, "b": 3.0}` | Valid |
| 2 | What is the sum of 265 and 345? | `fn_add_numbers` | `{"a": 265.0, "b": 345.0}` | Valid |
| 3 | Greet shrek | `fn_greet` | `{"name": "shrek"}` | Valid |
| 4 | Greet john | `fn_greet` | `{"name": "john"}` | Valid |
| 5 | Reverse the string 'hello' | `fn_reverse_string` | `{"s": "hello"}` | Valid |
| 6 | Reverse the string 'world' | `fn_reverse_string` | `{"s": "world"}` | Valid |
| 7 | What is the square root of 16? | `fn_get_square_root` | `{"a": 16.0}` | Valid |
| 8 | Calculate the square root of 144 | `fn_get_square_root` | `{"a": 144.0}` | Valid |
| 9 | Replace numbers in "..." with "NUMBERS" | `fn_substitute_string_with_regex` | `{"regex": "\\d+", "replacement": "NUMBERS", ...}` | Valid |
| 10 | Replace vowels with asterisks | `fn_substitute_string_with_regex` | `{"regex": "[aeiouAEIOU]", "replacement": "*", ...}` | Valid |
| 11 | Substitute 'cat' with 'dog' | `fn_substitute_string_with_regex` | `{"regex": "\\bcat\\b", "replacement": "dog", ...}` | Valid |

Type coercion also held up across the set: every numeric argument was emitted matching the schema's declared `number` type (e.g. `2.0` rather than `2`), and `model_dump()` confirmed each parsed dictionary against its dynamic validator without a single `ValidationError` on the public set.

### Speed
Per-step cost is dominated by two things: the forward pass through `Qwen/Qwen3-0.6B` (model-bound, identical with or without the tracker) and the masking pass over the ~151,000-entry vocabulary (`O(V)` per step, see [Algorithm Explanation](#algorithm-explanation)). Because masking only does constant-time prefix/suffix string comparisons per vocabulary entry — no backtracking regex, no tree search — its contribution to total latency stays small and flat relative to the model's own inference cost, and it does not grow as the JSON output gets longer or more deeply nested.

The double termination check (`end_condition`'s JSON-parse attempt, plus the forced `EOS_TOKEN_ID` in the `CLOSED` state) also means generation reliably stops as soon as the object is complete rather than running to `max_new_tokens` on every prompt, which keeps average wall-clock time per test close to the actual length of the JSON needed rather than the worst case.

### Reliability
The reliability story is the central claim of the project: where unconstrained generation from a model this size is reported to produce valid JSON only a fraction of the time, this pipeline produces 100% structurally valid output across every run, including outside the 11 public tests, because validity is enforced by the decoding grammar rather than hoped for from the model's training distribution. The only failure mode left structurally possible is a *semantic* one (the right shape, wrong content) — never a malformed or unparsable result — and that residual risk is caught by the post-generation Pydantic validation pass in `main()`, which logs a `validation_error` via `error_handling.json` rather than silently emitting bad data.

---

## Challenges Faced

- **String alignment under token escapes.** Strings look different in raw text than in decoded tokens — escaped quotes, carriage returns, and Unicode all shift offsets between the two representations. Slicing by raw character length against token boundaries produced off-by-one errors at string edges. The fix was computing `decoded_prefix` from the *decoded* token sequence rather than the original raw string, so all length comparisons in `_get_allowed_token_ids` happen in the same coordinate space the model actually operates in.
- **Greedy regular expression patterns.** Left unconstrained, the model would reliably open `regex` parameter values with a wildcard like `.*`, presumably because that's a common way regex literals start in its training data, even when the prompt asked for a much narrower pattern. The fix is structural rather than a prompt tweak: at the very first character of any `regex`-named value, tokens starting with a wildcard character are excluded from the allowed set entirely (`_get_allowed_token_ids`'s `VALUE_PARTIAL` branch and the equivalent check in `_is_valid_value_prefix`), so the model is never offered that option in the first place.
- **Re-deriving state without drift.** An earlier approach tracked JSON state incrementally as tokens arrived, which made it possible for the tracker's notion of "current key" or "current state" to fall out of sync with what had actually been decoded, especially around partially-formed tokens. Switching `_scan_parameters` to re-parse the full parameter string from scratch on every step removed that class of bug at the cost of repeating work that's cheap relative to the model's forward pass.

---

## Testing Strategy

Validation happens at three layers, each catching a different class of error:

1. **End-to-end generation tests against the public set.** `data/input/function_calling_tests.json` holds 11 natural-language prompts, one per available function (plus multiple phrasings per function to check robustness to wording). `make run` drives all of them through the full pipeline — prompt composition, constrained generation, JSON parsing, Pydantic validation — and writes results to `data/output/function_calls.json` for inspection. `make run-edge` runs the same pipeline against a separate edge-case input file to probe behavior the happy-path public set doesn't exercise (ambiguous phrasing, unusual argument formats, multiple plausible functions for one prompt).
2. **Structural correctness by construction, checked via `end_condition`.** Rather than testing for valid JSON after the fact, every generated response is validated *during* generation: `end_condition` attempts `json.loads()` on the in-progress text at every step, and the `CLOSED` state's `EOS_TOKEN_ID` forcing is a second independent confirmation that the object closed correctly. A test failing here would indicate a bug in the grammar logic itself, not in the model's output.
3. **Semantic and type validation post-generation.** In `main()`, every parsed response is checked against its function's compiled Pydantic validator and coerced via `model_dump()`. This is the layer that would catch a function call that's syntactically perfect JSON but semantically wrong (e.g. a string where a number was expected) — something the decoding grammar alone can't judge, since the grammar only knows the type the schema *declares*, not whether the model picked the *right* function for the prompt's actual intent.

Static analysis backs these runtime checks: `make lint` runs `flake8` for style and `mypy` (with `--warn-return-any`, `--disallow-untyped-defs`, and `--check-untyped-defs`) for type-correctness across `src`, and `make lint-strict` runs the same `mypy` pass in fully strict mode for a tighter bar during active development.

---

## Example Usage

**Set up the environment** (creates the `uv` virtual environment and installs `pydantic`, `numpy`, and the rest of the dependencies from `pyproject.toml`):
```bash
make install
```

**Run the default pipeline** against the bundled public test prompts (`data/input/function_calling_tests.json`), using the bundled function catalog (`data/input/functions_definition.json`):
```bash
make run
```
This prints the title screen, loads the model and tokenizer, then for each prompt shows a `GENERATION_TEST_NN` banner with the live token-by-token output as it's generated, e.g.:
```
============================== GENERATION_TEST_00 ===============================
PROMPT:  What is the sum of 2 and 3?
ANSWER: {"prompt":"What is the sum of 2 and 3?","name":"fn_add_numbers","parameters":{"a":2.0,"b":3.0}}
```
Final results for the whole run are written to `data/output/function_calls.json`.

**Run with fully custom inputs** — your own function catalog, your own prompts, and a custom output path — by calling the module directly instead of going through `make`:
```bash
uv run python -m src \
  --functions_definition path/to/my_functions.json \
  --input path/to/my_prompts.json \
  --output path/to/my_results.json
```
Each of the three flags is optional and falls back to the bundled defaults (`data/input/functions_definition.json`, `data/input/function_calling_tests.json`, `data/output/function_calls.json`) if omitted.

**Clean generated caches and artifacts:**
```bash
make clean
```

**Run static analysis** before committing changes:
```bash
make lint          # flake8 + mypy (lenient mode)
make lint-strict    # flake8 + mypy --strict
```

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

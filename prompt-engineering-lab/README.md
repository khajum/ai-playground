
# Prompt Engineering Lab

A starter lab for experimenting with prompt patterns, evaluation, and reproducible experiments.

## Structure
```
llm-lab/prompt-engineering-lab/
├─ README.md
├─ prompts/
│  ├─ zero_shot.json
│  ├─ few_shot.json
│  ├─ chain_of_thought.json
│  ├─ react.json
│  └─ constraints.json
├─ data/
│  └─ tasks.jsonl
├─ eval/
│  └─ harness.py
└─ run_eval.py
```

## Usage
Create your environment (see `requirements.txt` or `environment.yml`), then run:

```bash
# Zero-shot
python llm-lab/prompt-engineering-lab/run_eval.py --prompt prompts/zero_shot.json

# Few-shot
python llm-lab/prompt-engineering-lab/run_eval.py --prompt prompts/few_shot.json

# Chain-of-thought
python llm-lab/prompt-engineering-lab/run_eval.py --prompt prompts/chain_of_thought.json

# ReAct
python llm-lab/prompt-engineering-lab/run_eval.py --prompt prompts/react.json

# Persona & constraints
python llm-lab/prompt-engineering-lab/run_eval.py --prompt prompts/constraints.json
```

Outputs are written to `llm-lab/prompt-engineering-lab/eval/results.json` with per-task metrics:
- `exact_match` — exact string match (for QA/classification)
- `substr_match` — reference appears in prediction
- `rouge1_like` — unigram recall (toy ROUGE-1 approximation)

## Plug in a Real Model
Replace the `MockModel` in `eval/harness.py` with your provider.
Examples:
- **Hugging Face Transformers**: load `AutoModelForCausalLM` and `AutoTokenizer` with `generate()`.
- **Azure/OpenAI**: call your deployed endpoint; remember to implement rate limiting and retries.

## Tips
- Keep prompts small and explicit; add **constraints** for format and length.
- Use **few-shot** examples to lock the output format.
- Evaluate prompts per task type; avoid mixing objectives in one run.

*By: Ram Limbu — Last updated: 2025-12-27*

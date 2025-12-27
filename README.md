
# 🧠 AI Engineering Roadmap & LLM Lab

> A comprehensive, hands-on learning and building repo for AI Engineering—covering ML fundamentals, Deep Learning, Transformers, LLMs, Prompt Engineering, Paper Implementations, Experiments, and Capstone Projects.
>
> By: **Ram Limbu**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Frameworks](https://img.shields.io/badge/Frameworks-PyTorch%20%7C%20TensorFlow%20%7C%20HF%20Transformers-orange.svg)](https://pytorch.org/)

---

## 📚 What’s Inside
- **Roadmap** – A structured path from fundamentals → DL → LLMs → production.
- **Exercises & Solutions** – Progressive, hands-on practice across ML/AI topics.
- **LLM Lab** – Prompt engineering, finetuning, evaluation, deployment.
- **Paper Implementations** – Reproducible, annotated implementations of key papers.
- **Capstone Projects** – End-to-end projects showcasing real-world problem solving.
- **MLOps & Systems** – Data pipelines, evaluation, monitoring, and packaging.

---

## 🚀 Quick Start

### 1) Create Environment
```bash
# Using conda
conda env create -f environment.yml
conda activate ai-roadmap

# Or using pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Run a Demo
```bash
# Train a simple classifier (example)
make train            # or: python scripts/train.py --config configs/ml_basics.yaml

# Evaluate
make evaluate         # or: python scripts/evaluate.py --task llm_eval

# Serve an LLM endpoint
make serve            # or: uvicorn scripts.serve:app --reload
```

> Tip: See **[`docs/roadmap.md`](docs/roadmap.md)** for the full learning plan, deliverables, and checklists.

---

## 🗂 Repository Structure
```text
ai-playground /
├─ README.md
├─ .gitignore
├─ environment.yml
├─ requirements.txt
├─ Makefile
├─ docs/
│  ├─ roadmap.md
│  ├─ references.md
│  └─ architecture-notes/
├─ notebooks/
│  ├─ 01_ml_basics.ipynb
│  ├─ 02_deep_learning.ipynb
│  ├─ 03_transformers.ipynb
│  └─ 04_llm_eval.ipynb
├─ exercises/
│  ├─ ml-from-scratch/
│  ├─ deep-learning-playground/
│  └─ reinforcement-learning-gym/
├─ llm-lab/
│  ├─ prompt-engineering-lab/
│  ├─ finetune/
│  ├─ inference/
│  ├─ evaluation/
│  └─ deployment/
├─ paper-implementations/
│  ├─ attention-is-all-you-need/
│  ├─ diffusion-basics/
│  └─ retrieval-augmented-generation/
├─ capstones/
│  ├─ llm-finetuning-hub/
│  ├─ generative-ai-showcase/
│  └─ ai-capstone-projects/
├─ scripts/
│  ├─ setup_env.sh
│  ├─ train.py
│  ├─ evaluate.py
│  └─ serve.py
└─ tests/
```

---

## 🧭 Learn by Building: Roadmap (Phases)
The full roadmap lives in **[`docs/roadmap.md`](docs/roadmap.md)**. Highlights:
- **Phase 0 — Foundations:** Python, math for ML, classic algorithms.
- **Phase 1 — Deep Learning:** PyTorch, CNN/RNN, experiment tracking.
- **Phase 2 — Transformers:** Attention, tokenization, paper implementation.
- **Phase 3 — LLMs in Practice:** HF ecosystem, prompts, RAG, evaluation.
- **Phase 4 — Finetuning & Optimization:** LoRA/QLoRA, quantization, inference.
- **Phase 5 — Production AI:** APIs, CI/CD, Docker, monitoring, Responsible AI.

---

## 🧪 Modules & Demos
- `llm-lab/prompt-engineering-lab/` – Prompt patterns (Zero/Few-shot, CoT, ReAct) + eval harness.
- `llm-lab/finetune/` – SFT with PEFT (LoRA/QLoRA), configs, scripts.
- `llm-lab/inference/` – Efficient CPU/GPU pipelines, quantized inference.
- `llm-lab/evaluation/` – Metrics: accuracy, ROUGE/BLEU, toxicity, hallucinations.
- `llm-lab/deployment/` – FastAPI service, Dockerfiles, `compose.yml`.

---

## ⚙️ Makefile Commands
```make
.PHONY: setup train evaluate serve lint test

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

train:
	python scripts/train.py --config configs/train.yaml

evaluate:
	python scripts/evaluate.py --task llm_eval

serve:
	uvicorn scripts.serve:app --host 0.0.0.0 --port 8000 --reload

lint:
	ruff . && black --check .

test:
	pytest -q
```

---

## 🔐 Responsible AI
- Document intended use, limitations, and safety controls (model cards).
- Evaluate toxicity, bias, and hallucination rates; add guardrails.
- Respect privacy, compliance, and safe failure modes.

---

## 🤝 Contributing
1. Fork & create a feature branch: `feat/<module-name>`
2. Add tests and docs for new modules
3. Run `make lint && make test`
4. Open a PR with a clear description, screenshots, and benchmarks

---

## 📦 License
MIT — see [`LICENSE`](./LICENSE).

---

## 🙌 Acknowledgements
This repo leverages the open-source ecosystem (PyTorch, Hugging Face Transformers/Datasets, PEFT/TRL, FastAPI) and community best practices.

---

*By: Ram Limbu — Last updated: 2025-12-27*

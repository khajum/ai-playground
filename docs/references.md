
# 📚 References: AI Engineering, Transformers, LLMs, RAG & MLOps

## 1) Foundations (Python, Math, ML)
- **Python**
  - [Python Docs](https://docs.python.org/3/)
  - [Effective Python (book)](https://effectivepython.com/)
- **Math for ML**
  - [The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)
  - [Khan Academy: Linear Algebra](https://www.khanacademy.org/math/linear-algebra)
- **ML Fundamentals**
  - [ISLR (Intro to Statistical Learning)](https://www.statlearning.com/)
  - [Hands-On Machine Learning (book)](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125974/)

---

## 2) Deep Learning
- [Deep Learning (Goodfellow, Bengio, Courville)](https://www.deeplearningbook.org/)
- [Stanford CS231n](http://cs231n.stanford.edu/)
- [PyTorch Docs](https://pytorch.org/docs/stable/index.html)
- [MLflow](https://mlflow.org/docs/latest/index.html)
- [Weights & Biases](https://docs.wandb.ai/)

---

## 3) Transformers & Tokenization
- [Attention Is All You Need (paper)](https://arxiv.org/abs/1706.03762)
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [SentencePiece](https://github.com/google/sentencepiece)

---

## 4) Hugging Face Ecosystem
- [Transformers](https://huggingface.co/docs/transformers/index)
- [Datasets](https://huggingface.co/docs/datasets)
- [PEFT](https://huggingface.co/docs/peft/index)
- [TRL](https://huggingface.co/docs/trl/index)
- [Evaluate](https://huggingface.co/docs/evaluate/index)

---

## 5) LLMs: Training, Finetuning, Inference
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- [Chinchilla: Compute-Optimal LLMs](https://arxiv.org/abs/2203.15556)
- [LoRA (paper)](https://arxiv.org/abs/2106.09685)
- [QLoRA (paper)](https://arxiv.org/abs/2305.14314)
- [vLLM](https://vllm.readthedocs.io/)
- [Text Generation Inference](https://huggingface.co/docs/text-generation-inference/index)

---

## 6) Prompt Engineering & Evaluation
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [OpenAI Cookbook](https://github.com/openai/openai-cookbook)
- [BLEU](https://www.aclweb.org/anthology/P02-1040.pdf)
- [ROUGE](https://www.aclweb.org/anthology/W04-1013.pdf)

---

## 7) Retrieval Augmented Generation (RAG)
- [RAG (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Milvus](https://milvus.io/docs)

---

## 8) Responsible AI & Safety
- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)
- [Anthropic’s Constitutional AI](https://arxiv.org/abs/2304.10244)

---

## 9) MLOps, Deployment & Observability
- [FastAPI](https://fastapi.tiangolo.com/)
- [Docker](https://docs.docker.com/)
- [Kubernetes](https://kubernetes.io/docs/home/)
- [Prometheus](https://prometheus.io/docs/introduction/overview/)

---

## 10) Datasets
- [Hugging Face Hub](https://huggingface.co/datasets)
- [The Pile](https://pile.eleuther.ai/)

---

## 11) Paper Reading & Reproducibility
- [arXiv](https://arxiv.org/)
- [Papers with Code](https://paperswithcode.com/)

---

# 🧪 Practical Code Examples

Below are minimal, runnable examples to pair with the references. They are designed to be copy-paste starters.

## Foundations: NumPy/Pandas
```python
import numpy as np
import pandas as pd
np.random.seed(42)
data = pd.DataFrame({"age": np.random.randint(18, 60, size=8),"salary": np.random.randint(30000, 120000, size=8)})
data["salary_per_age"] = data["salary"] / data["age"]
print(data)

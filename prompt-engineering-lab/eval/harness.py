"""Prompt Engineering Lab Evaluation Harness
- Loads tasks from data/tasks.jsonl
- Loads prompt template from prompts/*.json
- Renders messages and queries a mock model (replace with your provider)
- Computes simple metrics (exact match, substring, unigram overlap)
"""
import os, json, argparse, re
from typing import Dict, Any, List

# ---- Simple templating ----
def render(template: str, variables: Dict[str, Any]) -> str:
    out = template
    for k, v in variables.items():
        out = out.replace('{{' + k + '}}', str(v))
    return out

# ---- Mock model ----
class MockModel:
    """A deterministic mock that produces simple outputs for demo purposes.
    Replace with real providers (OpenAI, Azure, HF) in providers/*.py.
    """
    def generate(self, system: str, user: str) -> str:
        # Tiny heuristic-based outputs for demo
        if 'Classify the sentiment' in user:
            text = user.lower()
            if any(w in text for w in ['love', 'amazing', 'great', 'excellent']):
                return 'Positive'
            if any(w in text for w in ['terrible', 'bad', 'poor']):
                return 'Negative'
            return 'Neutral'
        if 'Summarize' in user:
            # Return 2 bullets by splitting sentences
            bullets = re.split(r'[.!?]\s+', user)
            bullets = [b.strip() for b in bullets if b.strip()][:2]
            return '\n'.join(f'- {b[:100]}' for b in bullets)
        if 'UNESCO World Heritage' in user:
            return 'Pashupatinath Temple'
        if 'Answer:' in system or "Let's think step by step" in user:
            return 'Step 1: 90/3 = 30\nAnswer: NPR 30'
        return 'I am a mock model response.'

# ---- Metrics ----

def normalize(s: str) -> str:
    return re.sub(r'\s+', ' ', s.strip().lower())


def exact_match(pred: str, ref: str) -> float:
    if ref is None:
        return float('nan')
    return 1.0 if normalize(pred) == normalize(ref) else 0.0


def substr_match(pred: str, ref: str) -> float:
    if ref is None:
        return float('nan')
    return 1.0 if normalize(ref) in normalize(pred) else 0.0


def rouge1_like(pred: str, ref: str) -> float:
    if ref is None:
        return float('nan')
    P = set(normalize(pred).split())
    R = set(normalize(ref).split())
    if not R:
        return 0.0
    overlap = len(P & R)
    return overlap / len(R)

# ---- Runner ----

def run(prompt_file: str, tasks_path: str) -> Dict[str, Any]:
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompt = json.load(f)
    system = prompt.get('system', '')
    user_tmpl = prompt.get('user_template', '{{instruction}}\n\nInput:\n{{input}}')
    examples = prompt.get('examples', [])

    tasks: List[Dict[str, Any]] = []
    with open(tasks_path, 'r', encoding='utf-8') as f:
        for line in f:
            tasks.append(json.loads(line))

    model = MockModel()
    results = []
    for t in tasks:
        # Build user message
        user_msg = ''
        if examples:
            for ex in examples:
                user_msg += f"Example:\nUser: {ex['user']}\nAssistant: {ex['assistant']}\n\n"
        user_msg += render(user_tmpl, {"instruction": t['instruction'], "input": t['input']})

        pred = model.generate(system, user_msg)
        res = {
            'task_id': t['task_id'],
            'type': t['type'],
            'prediction': pred,
            'expected': t.get('expected'),
        }
        # Metrics
        res['exact_match'] = exact_match(pred, t.get('expected'))
        res['substr_match'] = substr_match(pred, t.get('expected'))
        res['rouge1_like'] = rouge1_like(pred, t.get('expected'))
        results.append(res)

    return {
        'prompt': prompt.get('name', os.path.basename(prompt_file)),
        'n_tasks': len(results),
        'results': results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompt', default='prompts/zero_shot.json')
    ap.add_argument('--tasks', default='data/tasks.jsonl')
    ap.add_argument('--out', default='eval/results.json')
    args = ap.parse_args()

    out = run(args.prompt, args.tasks)
    os.makedirs('eval', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Wrote results to {args.out}. {out['n_tasks']} tasks evaluated with prompt '{out['prompt']}'.")

if __name__ == '__main__':
    main()

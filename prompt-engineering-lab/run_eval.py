
#!/usr/bin/env python3
"""Convenience runner for the prompt engineering lab.
Usage examples:
  python llm-lab/prompt-engineering-lab/run_eval.py --prompt prompts/zero_shot.json
  python llm-lab/prompt-engineering-lab/run_eval.py --prompt prompts/few_shot.json
"""
import os, sys, argparse, subprocess

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompt', default='prompts/zero_shot.json')
    ap.add_argument('--tasks', default='data/tasks.jsonl')
    ap.add_argument('--out', default='eval/results.json')
    args = ap.parse_args()

    harness = os.path.join('llm-lab','prompt-engineering-lab','eval','harness.py')
    cmd = [sys.executable, harness, '--prompt', os.path.join('llm-lab','prompt-engineering-lab', args.prompt), '--tasks', os.path.join('llm-lab','prompt-engineering-lab', args.tasks), '--out', os.path.join('llm-lab','prompt-engineering-lab', args.out)]
    print('Running:', ' '.join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    main()

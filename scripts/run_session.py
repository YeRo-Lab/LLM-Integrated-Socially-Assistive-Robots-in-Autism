from __future__ import annotations

import argparse
from pathlib import Path
from pepper_llm_mvp.interaction.session_manager import SessionManager


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True)
    parser.add_argument('--condition', default='constrained_llm', choices=['scripted', 'unconstrained_llm', 'constrained_llm'])
    parser.add_argument('--response', default=None, help='Optional scripted user response for deterministic testing')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    manager = SessionManager(project_root=project_root, condition=args.condition, scripted_response=args.response, seed=args.seed)
    out = manager.run_task(args.task)
    print(f'Wrote session to {out}')


if __name__ == '__main__':
    main()
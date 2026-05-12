from __future__ import annotations

import argparse
from pathlib import Path
from pepper_llm_mvp.interaction.session_manager import SessionManager
from pepper_llm_mvp.evaluation.ablation import compute_ablation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True)
    parser.add_argument('--n_sessions', type=int, default=10)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    conditions = ['scripted', 'unconstrained_llm', 'constrained_llm']
    base_seed = 100
    for ci, condition in enumerate(conditions):
        for i in range(args.n_sessions):
            manager = SessionManager(project_root=project_root, condition=condition, seed=base_seed + ci * 1000 + i)
            manager.run_task(args.task)
    df = compute_ablation(project_root / 'data' / 'raw' / 'sessions')
    out_path = project_root / 'data' / 'processed' / 'tables' / 'table4.csv'
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))
    print(f'\nSaved {out_path}')


if __name__ == '__main__':
    main()
from __future__ import annotations

from pathlib import Path
from pepper_llm_mvp.analytics.metrics_latency import compute_latency_table


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sessions_dir = project_root / 'data' / 'raw' / 'sessions'
    out_dir = project_root / 'data' / 'processed' / 'tables'
    out_dir.mkdir(parents=True, exist_ok=True)
    df = compute_latency_table(sessions_dir)
    out_path = out_dir / 'table1.csv'
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))
    print(f'\nSaved {out_path}')


if __name__ == '__main__':
    main()

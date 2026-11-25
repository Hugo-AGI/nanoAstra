"""Success rates of a results directory: one row per task and a total, for each execution mode found.

    python scripts/summarize.py results/streaming
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    root = Path(sys.argv[1])
    modes: dict[str, dict[str, list[bool]]] = {}
    for path in sorted(root.rglob("episode_*/result.json")):
        result = json.loads(path.read_text())
        modes.setdefault(result["mode"], {}).setdefault(result["task"], []).append(bool(result["success"]))
    if not modes:
        raise SystemExit(f"no episode results under {root}")
    for mode, tasks in sorted(modes.items()):
        width = max(map(len, tasks))
        print(f"{mode}: {len(tasks)} tasks")
        for task, outcomes in sorted(tasks.items()):
            print(f"  {task:<{width}}  {sum(outcomes):3d} / {len(outcomes):<3d}")
        successes, episodes = sum(map(sum, tasks.values())), sum(map(len, tasks.values()))
        print(f"  {'total':<{width}}  {successes:3d} / {episodes:<3d}  {100 * successes / episodes:.1f}%")


if __name__ == "__main__":
    main()

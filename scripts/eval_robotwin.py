"""Evaluate nanoAstra on one RoboTwin 2.0 task.

    python scripts/eval_robotwin.py --task place_empty_cup --output results/streaming/place_empty_cup

The model is served by an OpenAI-compatible endpoint (scripts/serve.sh).  Every episode writes
``episode_<k>/`` under --output: the images sent each turn, trace.json (prompts, replies, step outcomes),
result.json and a video; ``results.json`` lists all episodes.  Finished episodes are skipped on a rerun.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
LAUNCH_DIR = Path.cwd()             # importing the RoboTwin environment changes the working directory

from nanoastra.agent import MODES, Agent  # noqa: E402
from nanoastra.client import ChatClient  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, help="a RoboTwin 2.0 task name (see nanoastra/robotwin/seeds)")
    parser.add_argument("--output", required=True, help="directory of this run")
    parser.add_argument("--mode", choices=MODES, default="streaming",
                        help="streaming (default): steps execute while the reply is written; blocking: the robot waits "
                             "for each whole reply; in both the simulator keeps running during inference. "
                             "paused: the simulator waits for the model, as in RoboTwin's own evaluation loop")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--start_episode", type=int, default=0)
    parser.add_argument("--api_base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="nanoAstra-9B", help="the served model name")
    parser.add_argument("--api_key", default=None, help="default: $OPENAI_API_KEY")
    parser.add_argument("--no_video", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from nanoastra.robotwin.env import RoboTwinEnv

    output = Path(args.output) if Path(args.output).is_absolute() else LAUNCH_DIR / args.output
    agent = Agent(ChatClient(args.api_base, args.model, args.api_key), args.mode)
    env = RoboTwinEnv(args.task, record_video=not args.no_video)
    for index in range(args.start_episode, args.start_episode + args.episodes):
        episode_dir = output / f"episode_{index:03d}"
        if (episode_dir / "result.json").is_file():
            continue
        try:
            episode = env.reset(index)
            logging.info("episode %d (seed %d): %s", index, episode.seed, episode.instruction)
            result = agent.run_episode(env, episode.instruction, episode_dir)
            if not args.no_video:
                env.save_video(episode_dir / "video.mp4")
        except Exception:  # noqa: BLE001
            logging.exception("episode %d did not finish; a rerun tries it again", index)
            continue
        finally:
            env.close_episode()
        record = {"task": args.task, "mode": args.mode, "episode": index, "seed": episode.seed,
                  "instruction": episode.instruction, **asdict(result)}
        (episode_dir / "result.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
        logging.info("episode %d: %s after %d turns", index, result.finished_reason, result.turns)

    episodes = [json.loads(path.read_text()) for path in sorted(output.glob("episode_*/result.json"))]
    successes = sum(episode["success"] for episode in episodes)
    summary = {"task": args.task, "mode": args.mode, "episodes": len(episodes), "successes": successes}
    (output / "results.json").write_text(json.dumps({**summary, "results": episodes}, indent=1), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

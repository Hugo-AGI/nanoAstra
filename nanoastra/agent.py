"""The closed loop: observe, ask the model for the next steps, execute them, repeat."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .client import ChatClient, encode_image, image_part, text_part
from .prompt import parse_reply, system_prompt, turn_core, turn_text
from .steps import execute_steps, parse_steps
from .streaming import RealTime

logger = logging.getLogger(__name__)

MODES = ("streaming", "blocking", "paused")
MAX_TURNS = {"streaming": 150, "blocking": 150, "paused": 45}
ACTION_SECONDS = 0.06       # RoboTwin's step limit counts policy actions, one per recorded frame: 15 physics steps
FAILED_REPLIES_ALLOWED = 3
DONE_LIMIT = 3


@dataclass
class EpisodeResult:
    success: bool
    finished_reason: str
    turns: int
    steps_used: int
    step_limit: int
    model_calls: int
    prompt_tokens: int
    completion_tokens: int
    wall_seconds: float
    sim_seconds: Optional[float] = None          # streaming / blocking: robot time of the episode
    success_seconds: Optional[float] = None      # streaming / blocking: robot time at which the task succeeded


class Agent:
    """``mode``:

    streaming  the simulator runs during inference and steps execute while the reply is being written
    blocking   the simulator runs during inference and the robot waits for each whole reply
    paused     the simulator waits for the model, as in RoboTwin's own evaluation loop

    With the simulator running, an episode may take as long as RoboTwin's step limit in robot time.
    """

    def __init__(self, client: ChatClient, mode: str = "streaming", max_turns: Optional[int] = None) -> None:
        if mode not in MODES:
            raise ValueError(f"mode {mode!r}; one of {MODES}")
        self.client, self.mode = client, mode
        self.max_turns = max_turns or MAX_TURNS[mode]

    def run_episode(self, env, instruction: str, out_dir: Optional[Path] = None) -> EpisodeResult:
        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
        clock = None
        if self.mode != "paused":
            clock = RealTime(env, self.mode == "streaming", time_limit_s=env.step_limit * ACTION_SECONDS)
        started = time.time()
        trace: list[dict] = []
        previous: Optional[dict] = None         # the last answered turn, kept in the conversation without images
        last_results: list[dict] = []
        calls = prompt_tokens = completion_tokens = 0
        failed_replies = done_replies = 0
        reason = "max_turns"
        system = None
        turn = 0
        for turn in range(1, self.max_turns + 1):
            if clock and clock.expired():
                reason, turn = "time_limit", turn - 1
                break
            observation = env.observe()
            state, views = observation["state"], observation["views"]
            jpegs = {view.name: encode_image(view.image) for view in views}
            if system is None:
                system = system_prompt(env.profile, env.workspace, [view.caption for view in views], env.image_notes)
                if out_dir is not None:
                    (out_dir / "system_prompt.txt").write_text(system, encoding="utf-8")
            if out_dir is not None:
                for name, jpeg in jpegs.items():
                    (out_dir / f"turn{turn:03d}_{name}.jpg").write_bytes(jpeg)
            text = turn_text(turn, instruction, state, last_results, list(jpegs))
            messages = [{"role": "system", "content": system}]
            if previous is not None:
                messages += [{"role": "user", "content": [text_part(previous["text"])]},
                             {"role": "assistant", "content": previous["reply"]}]
            images = [image_part(jpeg) for jpeg in jpegs.values()]
            messages.append({"role": "user", "content": images + [text_part(text)]})

            if clock:
                replies, reply, error = clock.call(self.client, messages)
            else:
                replies = [self.client.chat(messages)]
                reply, error = replies[0].text, replies[0].error
            calls += len(replies)
            prompt_tokens += sum(r.prompt_tokens for r in replies)
            completion_tokens += sum(r.completion_tokens for r in replies)
            if not error and reply.strip():
                previous = {"turn": turn, "text": turn_core(turn, instruction, state, last_results), "reply": reply}
            record = {"turn": turn, "prompt": text, "reply": reply, "state": state, "results": []}
            trace.append(record)
            if clock:
                record["sent_s"] = round(clock.sent, 3)

            steps = None
            if error:
                record["error"] = f"model call failed: {error}"
                last_results = [{"command": "(no command)", "ok": False,
                                 "note": "your previous reply could not be obtained; reply with the JSON object"}]
            else:
                try:
                    steps = parse_steps(parse_reply(reply))
                    failed_replies = 0
                except Exception as exc:  # noqa: BLE001
                    record["error"] = str(exc)
                    logger.warning("turn %d: could not parse the reply (%s)", turn, exc)
                    last_results = [{"command": "(unparseable reply)", "ok": False,
                                     "note": f"your reply was not a valid JSON object ({exc}); "
                                             "reply with the JSON object only"}]
            if steps is None:
                failed_replies += 1
                if clock:
                    clock.end_turn()
                if failed_replies > FAILED_REPLIES_ALLOWED:
                    reason = "model_error" if error else "parse_failure"
                    break
                if clock and clock.mark():
                    reason = "success" if clock.succeeded_in_time() else "time_limit"
                    break
                continue

            results, stop_requested = execute_steps(
                env, steps, gate=clock.gate(steps, turn == self.max_turns) if clock else None)
            if clock:
                if clock.cut_at:
                    record["cut_at"] = clock.cut_at
                    kept = clock.cut()
                    if previous is not None and previous["turn"] == turn:
                        previous["reply"] = kept
                clock.mark()
                clock.end_turn()
                record["end_s"] = round(clock.now(), 3)
            if not results:
                results = [{"command": "(no steps)", "kind": "empty", "ok": False, "note": ""}]
            record["results"] = results
            last_results = results

            if env.success:
                reason = "success" if not clock or clock.succeeded_in_time() else "time_limit"
                break
            if stop_requested:
                done_replies += 1
                if done_replies >= DONE_LIMIT:
                    reason = "stop_requested"
                    break
            else:
                done_replies = 0
            if env.budget_exhausted:
                reason = "step_limit"
                break

        if reason == "max_turns" and clock and clock.expired():
            reason = "time_limit"
        result = EpisodeResult(
            success=clock.succeeded_in_time() if clock else env.success, finished_reason=reason, turns=turn,
            steps_used=env.steps_used, step_limit=env.step_limit, model_calls=calls, prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens, wall_seconds=round(time.time() - started, 1),
            sim_seconds=round(clock.now(), 3) if clock else None,
            success_seconds=None if not clock or clock.success_at is None else round(clock.success_at, 3))
        if out_dir is not None:
            (out_dir / "trace.json").write_text(json.dumps(trace, indent=1), encoding="utf-8")
        return result

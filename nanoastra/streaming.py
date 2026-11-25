"""Execution on the robot's clock: blocking and streaming.

On a real robot the world does not stop while the model writes.  Here the simulator keeps running during
inference: the clock is physics time, and a reply takes as long as it would on a dedicated server
(``Latency``), independent of the machine the evaluation runs on.  Rendering and motion planning take no
simulated time.

blocking    the robot holds still until the whole reply has arrived, then executes its steps.
streaming   a step starts as soon as it has been written and the step before it has finished.  Once the
            reply is complete and at most ``PREFILL_STEPS`` steps are left, a new observation is sent; the
            remaining steps keep running and are prefilled as the start of the next reply, which the model
            continues.  The conversation records the turn as having sent only the steps that had run.

Each turn is computed as observe, request, execute.  The model sees nothing after its observation, so
executing every step no earlier than its arrival time is the same as running the request concurrently.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from .prompt import parse_reply
from .steps import Step

PREFILL_STEPS = 2
STEPS_KEY = '"steps"'
OPENING = {"stop": [STEPS_KEY], "include_stop_str_in_output": True}
CONTINUE = {"continue_final_message": True, "add_generation_prompt": False}
_COMMAND_END = re.compile(r'"\s*\]?\s*\}\s*$')


@dataclass
class Latency:
    """Seconds until output token ``j`` of a reply: nanoAstra-9B served by vLLM on a single H20."""
    first_token_s: float = 0.162
    per_prompt_token_s: float = 0.000089
    per_output_token_s: float = 0.00645

    def at(self, prompt_tokens: int, j: int) -> float:
        return self.first_token_s + self.per_prompt_token_s * prompt_tokens + self.per_output_token_s * max(0, j - 1)

    def whole(self, reply) -> float:
        return self.at(reply.prompt_tokens, max(1, reply.completion_tokens))


def step_spans(text: str) -> list[tuple[int, int]]:
    """Character offsets (start, end) of each element of the reply's "steps" list."""
    match = re.search(r'"steps"\s*:\s*\[', text)
    if not match:
        return []
    decoder, index, spans = json.JSONDecoder(), match.end(), []
    while True:
        while index < len(text) and text[index] in " \n\r\t,":
            index += 1
        if index >= len(text) or text[index] == "]":
            return spans
        try:
            _, end = decoder.raw_decode(text, index)
        except ValueError:
            return spans
        spans.append((index, end))
        index = end


def _steps(text: str) -> list:
    try:
        return parse_reply(text).get("steps") or []
    except (ValueError, AttributeError):
        return []


@dataclass
class Prefill:
    """The steps of a turn that were still to run when the next observation was sent."""
    header: str         # the reply up to its first step
    body: str           # those steps as written, up to but not including the last command's closing quote
    close: str          # the rest of the reply
    count: int


class RealTime:
    def __init__(self, env, streaming: bool, time_limit_s: float = 0.0, latency: Optional[Latency] = None) -> None:
        self.env, self.streaming, self.limit = env, streaming, time_limit_s
        self.latency = latency or Latency()
        self.zero = env.physics_steps
        self.prefill: Optional[Prefill] = None
        self.success_at: Optional[float] = None
        self.sent = self.replied = 0.0          # this turn: request sent, whole reply arrived
        self.prefilled = 0                      # this turn: steps carried over from the previous turn
        self.cut_at: Optional[int] = None       # this turn: the step before which the next observation is sent
        self.arrivals: list[float] = []
        self._text, self._spans = "", []

    def now(self) -> float:
        return (self.env.physics_steps - self.zero) * self.env.dt

    def expired(self) -> bool:
        return bool(self.limit) and self.now() >= self.limit

    def mark(self) -> bool:
        """Note when the task first succeeded; True once it has."""
        if self.env.success and self.success_at is None:
            self.success_at = self.now()
        return self.env.success

    def succeeded_in_time(self) -> bool:
        return self.success_at is not None and not (self.limit and self.success_at > self.limit)

    def wait_until(self, t: float) -> None:
        steps = int(round((t - self.now()) / self.env.dt))
        if steps > 0:
            self.env.idle(steps)
        self.mark()

    def call(self, client, messages: list[dict]):
        """Request this turn's reply.  Returns (replies, reply text, error).

        After a turn that left steps to run, the model first writes its opening (up to "steps"), then
        continues the opening followed by those steps.  They are already running, so if the continuation
        fails or rewrites them, the reply is taken to end with them.
        """
        self.sent, self.cut_at = self.now(), None
        prefill, self.prefill = self.prefill, None
        self.prefilled = prefill.count if prefill else 0
        if prefill is None:
            reply = client.chat(messages)
            self._arrive(reply.text, reply, self.sent, 0)
            return [reply], reply.text, reply.error
        replies, started, opening = [], self.sent, None
        while opening is None and len(replies) < 2:
            head = client.chat(messages, OPENING)
            replies.append(head)
            started += self.latency.whole(head)
            opening = head.text if not head.error and head.text.endswith(STEPS_KEY) else None
        header = prefill.header if opening is None else opening + prefill.header.split(STEPS_KEY, 1)[1]
        forced = header + prefill.body
        reply = client.chat(messages + [{"role": "assistant", "content": forced}], CONTINUE)
        replies.append(reply)
        text = forced + reply.text
        if reply.error or _steps(text)[:prefill.count] != _steps(forced + prefill.close):
            text = forced + prefill.close
        self._arrive(text, reply, started, len(forced), max(0, reply.prompt_tokens - head.prompt_tokens))
        return replies, text, None

    def _arrive(self, text: str, reply, started: float, skip: int, prompt_tokens: Optional[int] = None) -> None:
        """When each step of ``text`` arrives.  ``skip``: characters of ``text`` that precede the reply;
        ``prompt_tokens``: the tokens the server still has to read, if not the whole prompt."""
        prompt = reply.prompt_tokens if prompt_tokens is None else prompt_tokens
        self.replied = started + self.latency.at(prompt, max(1, reply.completion_tokens))
        self._text, self._spans, self.arrivals = text, step_spans(text), []
        for k, (_, end) in enumerate(self._spans):
            if k < self.prefilled:
                self.arrivals.append(self.sent)
            elif not self.streaming or not reply.chunks:
                self.arrivals.append(self.replied)
            else:
                tokens = next((tokens for chars, tokens in reply.chunks if chars >= end - skip), reply.chunks[-1][1])
                self.arrivals.append(started + self.latency.at(prompt, tokens))

    def end_turn(self) -> None:
        """The next observation waits for the whole reply: its text goes into the next request."""
        if not self.mark():
            self.wait_until(self.replied)

    def gate(self, steps: list, last_turn: bool = False):
        """Called before step ``i`` (1-based).  Ends the turn once the task has succeeded or the time limit
        has passed; ends it for the next observation when streaming allows; otherwise waits for the step."""
        def gate(i: int) -> bool:
            if self.mark() or self.expired():
                return True
            rest = steps[i - 1:]
            if (self.streaming and not last_turn and i > 1 and len(rest) <= PREFILL_STEPS
                    and self.now() >= self.replied and len(self._spans) == len(steps)
                    and all(isinstance(step, Step) for step in rest)
                    and _COMMAND_END.search(self._text[:self._spans[-1][1]])):
                self.cut_at = i
                return True
            self.wait_until(self.arrivals[i - 1] if i <= len(self.arrivals) else self.replied)
            return self.mark() or self.expired()
        return gate

    def cut(self) -> str:
        """Keep the steps from ``cut_at`` on for the next reply; return this turn's reply without them."""
        text, spans, i = self._text, self._spans, self.cut_at
        last_end = spans[-1][1]
        rest = text[spans[i - 1][0]:last_end]
        quote = rest.rstrip().rstrip("}").rstrip().rstrip("]").rstrip().rfind('"')
        self.prefill = Prefill(text[:spans[0][0]], rest[:quote], rest[quote:] + text[last_end:], len(spans) - i + 1)
        return text[:spans[i - 2][1]] + text[last_end:]

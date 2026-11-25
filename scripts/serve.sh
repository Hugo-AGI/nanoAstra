#!/bin/bash
# Serve nanoAstra-9B with vLLM as an OpenAI-compatible endpoint.
#   bash scripts/serve.sh <model directory or Hugging Face id> [port] [more vLLM options]
set -eu
MODEL=${1:?usage: serve.sh <model> [port] [more vLLM options]}
PORT=${2:-8000}
exec vllm serve "$MODEL" --served-model-name nanoAstra-9B --host 0.0.0.0 --port "$PORT" --seed 0 \
  --max-model-len 16384 --limit-mm-per-prompt '{"image":8,"video":0}' \
  --reasoning-parser qwen3 "${@:3}"

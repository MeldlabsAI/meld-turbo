#!/bin/zsh
# Start Qwen3.8-Flash-Next locally as an OpenAI-compatible server with the Meld Turbo settings:
# Q2_0 experts + dense projections re-encoded to Q8_0, MTP speculative decoding (3 draft tokens) over a
# 106K-token draft vocabulary, GPU sampling, model locked in RAM.
#
#   ./serve.sh                          # http://127.0.0.1:8080/v1  (web chat UI at http://127.0.0.1:8080)
#   PORT=8081 CTX=65536 NP=2 ./serve.sh # port, context, parallel slots
#   LISTEN=0.0.0.0 ./serve.sh           # reachable from the LAN (zsh sets $HOST to the hostname, hence LISTEN)
#
# Needs ~45 GB of free RAM and the GPU to itself.
set -e
home=${MELD_HOME:-$HOME/.meld-turbo}
bin=${MELD_BIN:-$home/llama.cpp/build/bin}
models=${MELD_MODELS:-$home/models}
model=$models/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-d8-00001-of-00002.gguf
mtp=$models/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
[ -f $model ] || { echo "missing $model (see README: download + tools/requant_dense.py)"; exit 1; }
[ -f $mtp ]   || { echo "missing $mtp (see README)"; exit 1; }

export MELD_DRAFT_VOCAB=${MELD_DRAFT_VOCAB-$home/draft_vocab.bin}

exec $bin/llama-server -m $model -ngl 99 -fa auto -c ${CTX:-131072} -np ${NP:-1} -n ${NPREDICT:-16384} \
  --host ${LISTEN:-127.0.0.1} --port ${PORT:-8080} \
  --alias "${MODEL_NAME:-Qwen3.8-125B-MoE},qwen3.8-flash-next" --jinja -bs -lm mlock \
  --spec-type draft-mtp -md $mtp --spec-draft-n-max ${NMAX:-3} ${=EXTRA}

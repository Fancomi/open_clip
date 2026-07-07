#!/usr/bin/env bash
# Run caption slot extraction with a local OpenAI-compatible VLM/LLM service,
# then collect slot frequencies and optionally overlay selected words on feature probes.
#
# Recommended first run:
#   bash analysis/run_slot_vlm.sh --limit 200
#
# Full Karpathy probe-aligned run:
#   bash analysis/run_slot_vlm.sh --limit 0 --metric both
#
# Important notes:
#   1. This script only sends captions to the model, not images. The task is lexical slot
#      extraction from short CLIP captions.
#   2. Default DATA is COCO karpathy_1cap.tsv because existing probe npz files are aligned
#      with COCO val2014/Karpathy order. If you switch DATA to train2014 or CC3M, use a
#      matching probe npz or skip overlay.
#   3. Default VLM ports are auto-detected from 8001-8008. Override with PORT=8001,8002.
#   4. Outputs are resumable: existing output rows are kept and only missing ids are called.
#   5. Overlay defaults to MIN_COUNT=10 for stable low-frequency words.
#      Set MIN_COUNT=1 only if you explicitly want singletons.
#   6. For large feature probes, use METRIC_MAX_POINTS/BG_MAX_POINTS to reduce overlay cost.
#   7. Temporary tests should use OUT_ROOT under /tmp. For formal experiments, set OUT_ROOT
#      to a persistent experiment directory.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="/root/paddlejob/workspace/env_run/penghaotian"

DATA="${DATA:-$BASE/datas/coco/annotations/karpathy_1cap.tsv}"
DATASET="${DATASET:-coco_val}"
OUT_ROOT="${OUT_ROOT:-/tmp/openclip_slot_vlm}"
LIMIT="${LIMIT:-200}"                 # 0 means all rows
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-}"                      # empty => auto-detect 8001-8008
MODEL="${MODEL:-}"                    # empty => first model from /v1/models
WORKERS="${WORKERS:-}"                # empty => number of ports
MAX_TOKENS="${MAX_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0}"
THINK="${THINK:-0}"                   # 0 disables Qwen/Gemma thinking in chat_template_kwargs

PROBE="${PROBE:-$ROOT/logs/20260507_baseline_cc3m/cc3m_pe_dinov3_leproj_muon_lr001_0429_1821/checkpoints/probe/step_007040.npz}"
RUN_OVERLAY="${RUN_OVERLAY:-1}"
SLOT_TYPES="${SLOT_TYPES:-nouns,verbs,adjectives,spatial_relations}"
TOP_K="${TOP_K:-5}"
BOTTOM_K="${BOTTOM_K:-5}"
MIN_COUNT="${MIN_COUNT:-10}"             # avoid low-count noisy overlay words
METRIC="${METRIC:-density}"           # density | curvature | both
KNN_K="${KNN_K:-50}"
MAX_POINTS_PER_WORD="${MAX_POINTS_PER_WORD:-200}"
METRIC_MAX_POINTS="${METRIC_MAX_POINTS:-0}"  # 0 means full feature set
BG_MAX_POINTS="${BG_MAX_POINTS:-0}"          # 0 means full background
SEED="${SEED:-0}"

usage() {
  cat <<'EOF'
Usage:
  bash analysis/run_slot_vlm.sh [options]

Options:
  --data PATH              TSV with filepath/caption columns
  --dataset NAME           Dataset name written into JSONL ids
  --out-root DIR           Output root directory
  --limit N                Number of rows to process; 0 means all rows
  --host HOST              VLM host, default 127.0.0.1
  --port PORTS             Comma-separated ports; default auto-detect 8001-8008
  --model MODEL_ID         OpenAI-compatible model id; default first /v1/models id
  --workers N              Concurrent workers; default number of ports
  --probe PATH             Probe npz for overlay
  --no-overlay             Only extract slots and collect frequencies
  --slot-types LIST        Comma-separated slot types for overlay
  --metric NAME            density | curvature | both
  --top-k N                High-frequency words per slot
  --bottom-k N             Low-frequency words per slot
  --min-count N            Minimum count for selected words, default 10
  --metric-max-points N    Compute density/curvature on subset; 0 means all
  --bg-max-points N        Draw background subset; 0 means all
  --help                   Show this help

Environment overrides are also supported, e.g.:
  PORT=8001,8002 WORKERS=2 OUT_ROOT=/tmp/slots bash analysis/run_slot_vlm.sh --limit 100

Output layout:
  OUT_ROOT/slot_requests.jsonl          Caption requests for VLM
  OUT_ROOT/slots.jsonl                  VLM slot extraction results
  OUT_ROOT/stats/slot_frequencies.json  Frequency dictionary
  OUT_ROOT/stats/*.png                  Frequency plots
  OUT_ROOT/overlay_min10/*.png          Feature overlay plots by default

Notes:
  - Use COCO karpathy_1cap.tsv when overlaying on existing 5000-row probe npz.
  - If DATA and PROBE are from different splits, overlay will fail or report low match rate.
  - The VLM output parser filters obvious function words and weak verbs in analysis/slots.py.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data) DATA="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --probe) PROBE="$2"; shift 2 ;;
    --no-overlay) RUN_OVERLAY=0; shift ;;
    --slot-types) SLOT_TYPES="$2"; shift 2 ;;
    --metric) METRIC="$2"; shift 2 ;;
    --top-k) TOP_K="$2"; shift 2 ;;
    --bottom-k) BOTTOM_K="$2"; shift 2 ;;
    --min-count) MIN_COUNT="$2"; shift 2 ;;
    --metric-max-points) METRIC_MAX_POINTS="$2"; shift 2 ;;
    --bg-max-points) BG_MAX_POINTS="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

mkdir -p "$OUT_ROOT"
REQ="$OUT_ROOT/slot_requests.jsonl"
SLOTS="$OUT_ROOT/slots.jsonl"
STATS_DIR="$OUT_ROOT/stats"
OVERLAY_DIR="$OUT_ROOT/overlay_min${MIN_COUNT}"

cd "$ROOT"

if [[ -z "$PORT" ]]; then
  ports=()
  for p in $(seq 8001 8008); do
    if python - "$HOST" "$p" <<'PY' >/dev/null 2>&1
import json, sys, urllib.request
host, port = sys.argv[1], sys.argv[2]
with urllib.request.urlopen(f'http://{host}:{port}/v1/models', timeout=1) as r:
    json.loads(r.read().decode('utf-8'))
PY
    then
      ports+=("$p")
    fi
  done
  if [[ ${#ports[@]} -eq 0 ]]; then
    echo "No VLM ports detected on $HOST:8001-8008. Set PORT=8001,8002 manually." >&2
    exit 1
  fi
  PORT="$(IFS=,; echo "${ports[*]}")"
fi

if [[ -z "$WORKERS" ]]; then
  WORKERS="$(tr ',' '\n' <<<"$PORT" | wc -l)"
fi

if [[ -z "$MODEL" ]]; then
  MODEL="$(python - "$HOST" "${PORT%%,*}" <<'PY'
import json, sys, urllib.request
host, port = sys.argv[1], sys.argv[2]
with urllib.request.urlopen(f'http://{host}:{port}/v1/models', timeout=5) as r:
    data = json.loads(r.read().decode('utf-8'))
models = data.get('data') or []
if not models:
    raise SystemExit('no model returned by /v1/models')
print(models[0]['id'])
PY
)"
fi

limit_args=()
if [[ "$LIMIT" != "0" ]]; then
  limit_args=(--limit "$LIMIT")
fi

echo "[slot-vlm] ROOT=$ROOT"
echo "[slot-vlm] DATA=$DATA"
echo "[slot-vlm] OUT_ROOT=$OUT_ROOT"
echo "[slot-vlm] HOST=$HOST PORT=$PORT WORKERS=$WORKERS MODEL=$MODEL THINK=$THINK"
echo "[slot-vlm] PROBE=$PROBE RUN_OVERLAY=$RUN_OVERLAY"

python -m analysis.run --mode make_slot_input \
  --data "$DATA" \
  --dataset "$DATASET" \
  --slot-out "$REQ" \
  "${limit_args[@]}"

python - "$REQ" "$SLOTS" "$HOST" "$PORT" "$MODEL" "$WORKERS" "$MAX_TOKENS" "$TEMPERATURE" "$THINK" "$SEED" <<'PY'
import json
import os
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from analysis.slot_prompts import format_slot_prompt
from analysis.slots import SLOT_TYPES, read_slot_jsonl

req_path, out_path, host, port_s, model, workers_s, max_tok_s, temp_s, think_s, seed_s = sys.argv[1:]
ports = [p.strip() for p in port_s.split(',') if p.strip()]
workers = int(workers_s)
max_tokens = int(max_tok_s)
temperature = float(temp_s)
think = bool(int(think_s))
random.seed(int(seed_s))

requests = [json.loads(line) for line in open(req_path, encoding='utf-8') if line.strip()]
done = {}
if os.path.exists(out_path):
    try:
        records, stats = read_slot_jsonl(out_path, strict=True)
        done = {r['id']: r for r in records}
    except Exception:
        bad_path = out_path + '.bad'
        os.replace(out_path, bad_path)
        print(f'[slot-vlm] existing slots jsonl is invalid; moved to {bad_path}', file=sys.stderr)

pending = [r for r in requests if r['id'] not in done]
print(f'[slot-vlm] requests={len(requests)} done={len(done)} pending={len(pending)}')


def parse_json(text):
    text = text.strip().replace('```json', '').replace('```', '').strip()
    try:
        return json.loads(text)
    except Exception:
        idx = text.find('{')
        if idx < 0:
            raise ValueError('no JSON object in response: ' + text[:200])
        obj, _ = json.JSONDecoder().raw_decode(text, idx)
        if not isinstance(obj, dict):
            raise ValueError('decoded response is not a JSON object')
        return obj


def call(i, rec):
    port = ports[i % len(ports)]
    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': format_slot_prompt(rec['caption'])}],
        'temperature': temperature,
        'max_tokens': max_tokens,
        'chat_template_kwargs': {'enable_thinking': think},
    }
    req = urllib.request.Request(
        f'http://{host}:{port}/v1/chat/completions',
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = json.loads(resp.read().decode('utf-8'))
    obj = parse_json(raw['choices'][0]['message']['content'])
    slots = {k: obj.get(k, []) for k in SLOT_TYPES}
    return i, {**rec, 'slots': slots}, time.time() - t0

if pending:
    with open(out_path, 'a', encoding='utf-8') as out:
        ok = bad = 0
        lat = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(call, i, rec): rec for i, rec in enumerate(pending)}
            for fut in as_completed(futs):
                rec = futs[fut]
                try:
                    _, item, dt = fut.result()
                    out.write(json.dumps(item, ensure_ascii=False) + '\n')
                    out.flush()
                    ok += 1
                    lat.append(dt)
                    if ok <= 8 or ok % 100 == 0:
                        print(f'[slot-vlm] ok={ok}/{len(pending)} id={item["id"]} {dt:.2f}s')
                except Exception as e:
                    bad += 1
                    print(f'[slot-vlm] BAD id={rec.get("id")} {type(e).__name__}: {e}', file=sys.stderr)
        print(json.dumps({
            'ok': ok,
            'bad': bad,
            'avg_latency': sum(lat) / len(lat) if lat else None,
            'max_latency': max(lat) if lat else None,
            'out': out_path,
        }, ensure_ascii=False))
else:
    print('[slot-vlm] no pending rows; reuse existing slots jsonl')
PY

python -m analysis.run --mode collect_slots \
  --slots "$SLOTS" \
  --out-dir "$STATS_DIR" \
  --top-n 40

if [[ "$RUN_OVERLAY" == "1" ]]; then
  if [[ ! -f "$PROBE" ]]; then
    echo "Probe npz not found: $PROBE" >&2
    exit 1
  fi
  metric_args=(--metric "$METRIC")
  python -m analysis.run --mode overlay_slots \
    --probe "$PROBE" \
    --slots "$SLOTS" \
    --out-dir "$OVERLAY_DIR" \
    --slot-types "$SLOT_TYPES" \
    --top-k "$TOP_K" \
    --bottom-k "$BOTTOM_K" \
    --min-count "$MIN_COUNT" \
    "${metric_args[@]}" \
    --k "$KNN_K" \
    --max-points-per-word "$MAX_POINTS_PER_WORD" \
    --metric-max-points "$METRIC_MAX_POINTS" \
    --background-max-points "$BG_MAX_POINTS" \
    --seed "$SEED" \
    --save-geometry-summary
fi

echo "[slot-vlm] done"
echo "[slot-vlm] requests: $REQ"
echo "[slot-vlm] slots   : $SLOTS"
echo "[slot-vlm] stats   : $STATS_DIR"
if [[ "$RUN_OVERLAY" == "1" ]]; then
  echo "[slot-vlm] overlay : $OVERLAY_DIR"
fi

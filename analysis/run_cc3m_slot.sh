#!/usr/bin/env bash
# CC3M slot analysis — end-to-end pipeline:
#   1. Extract N samples (image + caption) from CC3M WDS tarballs to disk
#   2. Generate PE-Core image feature probe (npz)
#   3. Run VLM slot extraction on CC3M captions
#   4. Collect slot frequencies + stats
#   5. Overlay selected words on feature probe (min_count=10)
#
# Usage:
#   bash analysis/run_cc3m_slot.sh                       # default 50000 samples
#   bash analysis/run_cc3m_slot.sh --limit 1000          # quick test
#   bash analysis/run_cc3m_slot.sh --no-overlay          # skip overlay
#   bash analysis/run_cc3m_slot.sh --no-extract          # skip if already extracted

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="/root/paddlejob/workspace/env_run/penghaotian"

source "$BASE/envs/dino/bin/activate"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

# ── Config ────────────────────────────────────────────────────────────────────
CC3M_WDS_DIR="${CC3M_WDS_DIR:-$BASE/datas/cc3m-wds}"
LIMIT="${LIMIT:-50000}"
SEED="${SEED:-42}"
OUT_ROOT="${OUT_ROOT:-}"
SAMPLE_DIR="${SAMPLE_DIR:-}"
TSV_PATH="${TSV_PATH:-}"
PROBE_PATH="${PROBE_PATH:-}"
PE_CORE_CKPT="${PE_CORE_CKPT:-$BASE/models/timm/PE-Core-B-16/open_clip_model.safetensors}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-}"
MODEL="${MODEL:-}"
WORKERS="${WORKERS:-}"
MAX_TOKENS="${MAX_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0}"
THINK="${THINK:-0}"

RUN_EXTRACT="${RUN_EXTRACT:-1}"
RUN_PROBE="${RUN_PROBE:-1}"
RUN_OVERLAY="${RUN_OVERLAY:-1}"
SLOT_TYPES="${SLOT_TYPES:-nouns,verbs,adjectives,spatial_relations}"
TOP_K="${TOP_K:-5}"
BOTTOM_K="${BOTTOM_K:-5}"
MIN_COUNT="${MIN_COUNT:-10}"
METRIC="${METRIC:-both}"
KNN_K="${KNN_K:-50}"
MAX_POINTS_PER_WORD="${MAX_POINTS_PER_WORD:-200}"
METRIC_MAX_POINTS="${METRIC_MAX_POINTS:-0}"
BG_MAX_POINTS="${BG_MAX_POINTS:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit) LIMIT="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --no-extract) RUN_EXTRACT=0; shift ;;
    --no-probe) RUN_PROBE=0; shift ;;
    --no-overlay) RUN_OVERLAY=0; shift ;;
    --metric) METRIC="$2"; shift 2 ;;
    --min-count) MIN_COUNT="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    *) echo "Unknown: $1" >&2; exit 1 ;;
  esac
done

OUT_ROOT="${OUT_ROOT:-$ROOT/analysis/outputs/slots/cc3m_${LIMIT}}"
SAMPLE_DIR="${SAMPLE_DIR:-$OUT_ROOT/sample}"
TSV_PATH="${TSV_PATH:-$SAMPLE_DIR/cc3m_sample.tsv}"
PROBE_PATH="${PROBE_PATH:-$SAMPLE_DIR/probe_pe_core.npz}"

mkdir -p "$OUT_ROOT" "$SAMPLE_DIR"
REQ="$OUT_ROOT/slot_requests.jsonl"
SLOTS="$OUT_ROOT/slots.jsonl"
STATS_DIR="$OUT_ROOT/stats"

echo "[cc3m-slot] ROOT=$ROOT"
echo "[cc3m-slot] CC3M_WDS_DIR=$CC3M_WDS_DIR  LIMIT=$LIMIT  SEED=$SEED"
echo "[cc3m-slot] SAMPLE_DIR=$SAMPLE_DIR"
echo "[cc3m-slot] OUT_ROOT=$OUT_ROOT"
echo "[cc3m-slot] RUN_EXTRACT=$RUN_EXTRACT  RUN_PROBE=$RUN_PROBE  RUN_OVERLAY=$RUN_OVERLAY"

# ── Step 1: Extract CC3M samples to disk ─────────────────────────────────────
if [[ "$RUN_EXTRACT" == "1" ]]; then
  if [[ -f "$TSV_PATH" ]] && [[ "$(wc -l < "$TSV_PATH")" -gt "$LIMIT" ]]; then
    echo "[cc3m-slot] TSV already has enough rows — skipping extraction"
  else
    echo "[cc3m-slot] Extracting $LIMIT samples from CC3M WDS..."
    python - "$CC3M_WDS_DIR" "$SAMPLE_DIR" "$TSV_PATH" "$LIMIT" "$SEED" <<'PY'
import os, sys, tarfile, random, io, shutil
from pathlib import Path

wds_dir, sample_dir, tsv_path, limit_s, seed_s = sys.argv[1:]
limit = int(limit_s)
seed  = int(seed_s)

img_dir = os.path.join(sample_dir, 'images')
os.makedirs(img_dir, exist_ok=True)

tars = sorted(Path(wds_dir).glob('cc3m-train-*.tar'))
assert tars, f'No cc3m-train-*.tar found in {wds_dir}'

# Deterministic shuffle of tar list
rng = random.Random(seed)
rng.shuffle(tars)

collected = []
seen_keys = set()

for tar_path in tars:
    if len(collected) >= limit:
        break
    try:
        with tarfile.open(tar_path) as tf:
            members = tf.getmembers()
            # Build key -> (jpg_member, txt_member) map
            key_map = {}
            for m in members:
                stem, ext = os.path.splitext(m.name)
                if ext.lower() in ('.jpg', '.jpeg'):
                    key_map.setdefault(stem, {})['img'] = m
                elif ext.lower() == '.txt':
                    key_map.setdefault(stem, {})['txt'] = m

            keys = sorted(k for k, v in key_map.items()
                          if 'img' in v and 'txt' in v and k not in seen_keys)
            rng.shuffle(keys)

            for key in keys:
                if len(collected) >= limit:
                    break
                v = key_map[key]
                caption = tf.extractfile(v['txt']).read().decode('utf-8', errors='replace').strip()
                if not caption:
                    continue
                # Save image
                out_img = os.path.join(img_dir, f'{Path(tar_path).stem}_{key}.jpg')
                if not os.path.exists(out_img):
                    img_data = tf.extractfile(v['img']).read()
                    with open(out_img, 'wb') as f:
                        f.write(img_data)
                collected.append((out_img, caption))
                seen_keys.add(key)
    except Exception as e:
        print(f'[extract] WARN skip {tar_path.name}: {e}', file=sys.stderr)

print(f'[extract] collected {len(collected)} samples')

# Write TSV
with open(tsv_path, 'w', encoding='utf-8') as f:
    f.write('filepath\tcaption\n')
    for path, cap in collected:
        f.write(f'{path}\t{cap}\n')
print(f'[extract] TSV -> {tsv_path}')
PY
  fi
else
  echo "[cc3m-slot] Skipping extraction (--no-extract)"
fi

TSV_ROWS=$(wc -l < "$TSV_PATH")
echo "[cc3m-slot] TSV rows (incl. header): $TSV_ROWS"

# ── Step 2: Generate PE-Core image feature probe ──────────────────────────────
if [[ "$RUN_PROBE" == "1" ]]; then
  if [[ -f "$PROBE_PATH" ]]; then
    echo "[cc3m-slot] Probe already exists — skipping"
  else
    echo "[cc3m-slot] Generating PE-Core probe from $TSV_PATH ..."
    python - "$TSV_PATH" "$PROBE_PATH" "$PE_CORE_CKPT" <<'PY'
import sys, os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
import numpy as np
import pandas as pd
import torch
import open_clip

tsv_path, probe_path, pe_ckpt = sys.argv[1:]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'[probe] device={device}')

# Load PE-Core pretrained
print(f'[probe] Loading PE-Core-B-16 from {pe_ckpt}...')
model, _, preproc = open_clip.create_model_and_transforms(
    'PE-Core-B-16', pretrained=pe_ckpt)
model = model.eval().to(device)
print('[probe] Model loaded')

# Load paths + captions
df = pd.read_csv(tsv_path, sep='\t')
paths = df['filepath'].tolist()
caps  = df['caption'].tolist() if 'caption' in df.columns else None
print(f'[probe] Extracting features for {len(paths)} images...')

from open_clip_train.probe_hook import extract_backbone_cls, extract_text_features

bb_cls, proj_cls = extract_backbone_cls(model, paths, preproc, device, batch_size=256)
txt_feats = extract_text_features(model, caps, device) if caps else None

os.makedirs(os.path.dirname(probe_path), exist_ok=True)
save_kw = dict(features=bb_cls, paths=np.array(paths))
if proj_cls is not None:
    save_kw['proj_features'] = proj_cls
if txt_feats is not None:
    save_kw['txt_features'] = txt_feats
np.savez_compressed(probe_path, **save_kw)
print(f'[probe] Saved {probe_path}  features={bb_cls.shape}'
      + (f'  proj={proj_cls.shape}' if proj_cls is not None else '')
      + (f'  txt={txt_feats.shape}' if txt_feats is not None else ''))
PY
  fi
else
  echo "[cc3m-slot] Skipping probe generation (--no-probe)"
fi

# ── Step 3: Build VLM request JSONL ──────────────────────────────────────────
REQ_ROWS_EXPECTED=$((TSV_ROWS - 1))
if [[ -f "$REQ" ]] && [[ "$(wc -l < "$REQ")" -eq "$REQ_ROWS_EXPECTED" ]]; then
  echo "[cc3m-slot] Requests already exist with $REQ_ROWS_EXPECTED rows — skipping"
else
  python -m analysis.run --mode make_slot_input \
    --data "$TSV_PATH" \
    --dataset cc3m \
    --slot-out "$REQ"
fi

# ── Step 4: Auto-detect VLM port ─────────────────────────────────────────────
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
    echo "[cc3m-slot] No VLM ports detected on $HOST:8001-8008; requests are ready at $REQ." >&2
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
print(data['data'][0]['id'])
PY
)"
fi

echo "[cc3m-slot] PORT=$PORT  WORKERS=$WORKERS  MODEL=$MODEL"

# ── Step 5: VLM slot extraction ───────────────────────────────────────────────
python - "$REQ" "$SLOTS" "$HOST" "$PORT" "$MODEL" "$WORKERS" "$MAX_TOKENS" "$TEMPERATURE" "$THINK" "$SEED" <<'PY'
import json, os, random, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from analysis.slot_prompts import format_slot_prompt
from analysis.slots import SLOT_TYPES, read_slot_jsonl

req_path, out_path, host, port_s, model, workers_s, max_tok_s, temp_s, think_s, seed_s = sys.argv[1:]
ports   = [p.strip() for p in port_s.split(',') if p.strip()]
workers = int(workers_s)
max_tokens  = int(max_tok_s)
temperature = float(temp_s)
think       = bool(int(think_s))
random.seed(int(seed_s))

requests = [json.loads(l) for l in open(req_path, encoding='utf-8') if l.strip()]
done = {}
if os.path.exists(out_path):
    try:
        records, _ = read_slot_jsonl(out_path, strict=True)
        done = {r['id']: r for r in records}
    except Exception:
        bad = out_path + '.bad'
        os.replace(out_path, bad)
        print(f'[slot] existing slots.jsonl invalid, moved to {bad}', file=sys.stderr)

pending = [r for r in requests if r['id'] not in done]
print(f'[slot] requests={len(requests)} done={len(done)} pending={len(pending)}')

def parse_json(text):
    text = text.strip().replace('```json','').replace('```','').strip()
    try:
        return json.loads(text)
    except Exception:
        idx = text.find('{')
        if idx < 0: raise ValueError('no JSON in: ' + text[:200])
        obj, _ = json.JSONDecoder().raw_decode(text, idx)
        return obj

def call(i, rec):
    port = ports[i % len(ports)]
    payload = {
        'model': model,
        'messages': [{'role':'user','content': format_slot_prompt(rec['caption'])}],
        'temperature': temperature, 'max_tokens': max_tokens,
        'chat_template_kwargs': {'enable_thinking': think},
    }
    req = urllib.request.Request(
        f'http://{host}:{port}/v1/chat/completions',
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type':'application/json'})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = json.loads(resp.read().decode('utf-8'))
    obj   = parse_json(raw['choices'][0]['message']['content'])
    slots = {k: obj.get(k, []) for k in SLOT_TYPES}
    return i, {**rec, 'slots': slots}, time.time() - t0

if pending:
    with open(out_path, 'a', encoding='utf-8') as out:
        ok = bad = 0; lat = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(call, i, rec): rec for i, rec in enumerate(pending)}
            for fut in as_completed(futs):
                rec = futs[fut]
                try:
                    _, item, dt = fut.result()
                    out.write(json.dumps(item, ensure_ascii=False) + '\n')
                    out.flush()
                    ok += 1; lat.append(dt)
                    if ok <= 8 or ok % 200 == 0:
                        print(f'[slot] ok={ok}/{len(pending)} id={item["id"]} {dt:.2f}s')
                except Exception as e:
                    bad += 1
                    print(f'[slot] BAD id={rec.get("id")} {type(e).__name__}: {e}', file=sys.stderr)
        print(json.dumps({'ok':ok,'bad':bad,
            'avg_latency': sum(lat)/len(lat) if lat else None,
            'max_latency': max(lat) if lat else None}))
else:
    print('[slot] no pending rows; reusing existing slots.jsonl')
PY

# ── Step 6: Collect slot frequencies ─────────────────────────────────────────
python -m analysis.run --mode collect_slots \
  --slots "$SLOTS" \
  --out-dir "$STATS_DIR" \
  --top-n 40

# ── Step 7: Overlay (min_count=10) ───────────────────────────────────────────
if [[ "$RUN_OVERLAY" == "1" ]]; then
  if [[ ! -f "$PROBE_PATH" ]]; then
    echo "[cc3m-slot] Probe not found: $PROBE_PATH — skipping overlay" >&2
  else
    OVERLAY_DIR="$OUT_ROOT/overlay_min10"
    echo "[cc3m-slot] Overlay min_count=10 -> $OVERLAY_DIR"
    python -m analysis.run --mode overlay_slots \
      --probe "$PROBE_PATH" \
      --slots "$SLOTS" \
      --out-dir "$OVERLAY_DIR" \
      --slot-types "$SLOT_TYPES" \
      --top-k "$TOP_K" \
      --bottom-k "$BOTTOM_K" \
      --min-count "$MIN_COUNT" \
      --metric "$METRIC" \
      --k "$KNN_K" \
      --max-points-per-word "$MAX_POINTS_PER_WORD" \
      --metric-max-points "$METRIC_MAX_POINTS" \
      --background-max-points "$BG_MAX_POINTS" \
      --seed "$SEED" \
      --save-geometry-summary
  fi
fi

echo ""
echo "[cc3m-slot] ===== DONE ====="
echo "[cc3m-slot] TSV     : $TSV_PATH"
echo "[cc3m-slot] Probe   : $PROBE_PATH"
echo "[cc3m-slot] Requests: $REQ"
echo "[cc3m-slot] Slots   : $SLOTS"
echo "[cc3m-slot] Stats   : $STATS_DIR"
if [[ "$RUN_OVERLAY" == "1" ]]; then
  echo "[cc3m-slot] Overlay : $OUT_ROOT/overlay_min10"
fi

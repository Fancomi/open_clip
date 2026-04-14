#!/usr/bin/env python3
"""
将 LLaVA-ReCap-CC3M parquet shards 转换为 webdataset tar 格式。

- 图片字节直接写入（无重编码）
- 增量运行：已存在的 tar 自动跳过
- 完成后输出 _stats.json 及 quick.sh 配置片段

用法:
  python3 scripts/convert_cc3m_to_wds.py
  python3 scripts/convert_cc3m_to_wds.py --workers 16
"""
import io
import json
import tarfile
import logging
import argparse
from pathlib import Path
from multiprocessing import Pool

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ── worker（module-level，multiprocessing 可 pickle）──────────────────────────

def _convert(task: tuple) -> tuple[str, int, int]:
    """转换单个 parquet shard → tar。返回 (shard_id, n_ok, n_err)。"""
    parquet_path, out_dir = Path(task[0]), Path(task[1])

    # "train-00000-of-00281" -> "00000"
    shard_id = parquet_path.stem.split("-")[1]
    tar_path = out_dir / f"{shard_id}.tar"
    tmp_path = out_dir / f"{shard_id}.tmp.tar"

    df = pd.read_parquet(parquet_path)
    n_ok = n_err = 0

    with tarfile.open(tmp_path, "w") as tf:
        for row in df.itertuples(index=False):
            try:
                img_bytes = row.image["bytes"]
                caption   = row.conversations[1]["value"]
                key       = str(row.id)

                for ext, data in ((".jpg", img_bytes), (".txt", caption.encode())):
                    buf  = io.BytesIO(data)
                    info = tarfile.TarInfo(key + ext)
                    info.size = len(data)
                    tf.addfile(info, buf)

                n_ok += 1
            except Exception:
                n_err += 1

    tmp_path.rename(tar_path)
    log.info(f"[{shard_id}]  {n_ok:>6,} ok  {n_err} err  →  {tar_path.name}")
    return shard_id, n_ok, n_err


# ── 聚合统计 ──────────────────────────────────────────────────────────────────

def write_stats(out_dir: Path, counts: dict[str, int]) -> dict:
    """counts: 本次转换结果 {shard_id: n_ok}，历史 shard 用固定值估算。"""
    tars  = sorted(out_dir.glob("?????.tar"))
    total = sum(counts.get(t.stem, 10170) for t in tars)
    last  = tars[-1].stem if tars else "00000"
    stats = {"num_shards": len(tars), "num_samples": total, "last_shard": last}
    (out_dir / "_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/root/paddlejob/workspace/env_run/penghaotian/datas/LLaVA-ReCap-CC3M/data")
    p.add_argument("--out-dir",  default="/root/paddlejob/workspace/env_run/penghaotian/datas/LLaVA-ReCap-CC3M/wds")
    p.add_argument("--workers",  type=int, default=8)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquets = sorted(data_dir.glob("train-*.parquet"))
    pending  = [p for p in parquets
                if not (out_dir / f"{p.stem.split('-')[1]}.tar").exists()]

    log.info(f"shards: {len(parquets)} total | {len(parquets)-len(pending)} done | {len(pending)} pending")

    counts: dict[str, int] = {}
    if pending:
        tasks = [(str(p), str(out_dir)) for p in pending]
        with Pool(args.workers) as pool:
            results = list(pool.imap_unordered(_convert, tasks))
        counts  = {sid: n_ok for sid, n_ok, _ in results}
        n_err   = sum(r[2] for r in results)
        if n_err:
            log.warning(f"total skipped rows (decode error): {n_err:,}")

    stats = write_stats(out_dir, counts)
    log.info(f"stats: {stats['num_shards']} shards, {stats['num_samples']:,} samples")

    print(f"\n=== quick.sh 配置片段 ===")
    print(f'CC3M="{out_dir}"')
    print(f'CC3M_TRAIN="${{CC3M}}/{{00000..{stats["last_shard"]}}}.tar"')
    print(f'CC3M_N_TRAIN={stats["num_samples"]}')


if __name__ == "__main__":
    main()

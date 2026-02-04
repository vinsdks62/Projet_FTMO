#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Grid runner robuste pour ftmo_crypto_pipeline.py

Objectif:
- Lancer une grille de backtests (TP, fee, slippage)
- Capturer et parser le bloc JSON "FINAL METRICS (copiable)"
- Sauvegarder chaque run dans results_runs/<tag> (pour éviter l'overwrite)
- Calculer en plus un Max Drawdown depuis equity.csv (si présent)
- Tout afficher dans le terminal (copiable)

Usage exemple:
python grid_runner.py \
  --start 2024-01-01 --end 2024-06-30 \
  --symbol BTC/USDT --exchange binance \
  --balance 10000 --daily-loss 0.05 --max-loss 0.10 \
  --sl 0.002 \
  --tps 0.0045,0.005,0.006,0.007 \
  --fees 0.0004,0.0006,0.0008 \
  --slips 0.0002,0.00035,0.0005
"""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime

FINAL_METRICS_MARKER = "FINAL METRICS (copiable)"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--symbol", required=True)
    p.add_argument("--exchange", required=True)
    p.add_argument("--balance", type=float, required=True)
    p.add_argument("--daily-loss", type=float, required=True)
    p.add_argument("--max-loss", type=float, required=True)
    p.add_argument("--sl", type=float, required=True)

    p.add_argument("--tps", required=True, help="liste csv ex: 0.0045,0.006")
    p.add_argument("--fees", required=True, help="liste csv ex: 0.0004,0.0008")
    p.add_argument("--slips", required=True, help="liste csv ex: 0.0002,0.0005")

    p.add_argument("--python", default=sys.executable, help="interpréteur python à utiliser")
    p.add_argument("--script", default="ftmo_crypto_pipeline.py", help="script pipeline")
    p.add_argument("--results-dir", default="results", help="dossier results généré par le pipeline")
    p.add_argument("--runs-dir", default="results_runs", help="dossier où stocker chaque run")
    return p.parse_args()


def split_floats(csv_list: str):
    out = []
    for x in csv_list.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(float(x))
    return out


def extract_final_metrics(stdout: str):
    """
    Extrait le JSON du bloc FINAL METRICS.

    Fix: le regex r"{.*}" est gourmand (greedy) et avale potentiellement
    plusieurs objets JSON / texte après le bloc, ce qui crée:
    json.decoder.JSONDecodeError: Extra data

    Solution: trouver le premier '{' après le marker puis utiliser
    JSONDecoder.raw_decode pour ne décoder que le premier objet JSON.
    """
    idx = stdout.find(FINAL_METRICS_MARKER)
    if idx < 0:
        return None

    tail = stdout[idx + len(FINAL_METRICS_MARKER):]
    j = tail.find("{")
    if j < 0:
        return None

    tail = tail[j:]
    dec = json.JSONDecoder()
    try:
        obj, _end = dec.raw_decode(tail)
        return obj
    except json.JSONDecodeError:
        # fallback: tenter un nettoyage simple (rare)
        cleaned = re.sub(r",\s*}", "}", tail)
        cleaned = re.sub(r",\s*]", "]", cleaned)
        try:
            obj, _end = dec.raw_decode(cleaned)
            return obj
        except json.JSONDecodeError:
            return None


def compute_max_drawdown_from_equity_csv(equity_csv_path: str):
    """
    Calcule MDD absolu et % à partir d'equity.csv
    Attend une colonne equity ou balance ou close_equity.
    Si fichier absent -> None
    """
    if not os.path.exists(equity_csv_path):
        return None

    with open(equity_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return None

    # trouver colonne equity
    candidates = ["equity", "balance", "close_equity", "account_equity"]
    col = None
    for c in candidates:
        if c in rows[0]:
            col = c
            break
    if col is None:
        return None

    eq = []
    for r in rows:
        try:
            eq.append(float(r[col]))
        except Exception:
            continue

    if len(eq) < 2:
        return None

    peak = eq[0]
    mdd_abs = 0.0
    mdd_pct = 0.0
    for v in eq:
        if v > peak:
            peak = v
        dd = peak - v
        if dd > mdd_abs:
            mdd_abs = dd
            mdd_pct = (dd / peak) if peak != 0 else 0.0

    return {"mdd_abs": mdd_abs, "mdd_pct": mdd_pct}


def safe_copytree(src_dir: str, dst_dir: str):
    os.makedirs(dst_dir, exist_ok=True)
    if not os.path.exists(src_dir):
        return
    for name in os.listdir(src_dir):
        s = os.path.join(src_dir, name)
        d = os.path.join(dst_dir, name)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)


def run_one(py, script, args_list):
    cmd = [py, script] + args_list
    p = subprocess.run(cmd, capture_output=True, text=True)
    stdout = p.stdout
    stderr = p.stderr
    return p.returncode, stdout, stderr, cmd


def main():
    a = parse_args()
    tps = split_floats(a.tps)
    fees = split_floats(a.fees)
    slips = split_floats(a.slips)

    os.makedirs(a.runs_dir, exist_ok=True)

    summary = []

    for tp in tps:
        for fee in fees:
            for slip in slips:
                tag = f"tp={tp}_fee={fee}_slip={slip}_{a.start}_to_{a.end}"
                tag = tag.replace("/", "_")
                outdir = os.path.join(a.runs_dir, tag)

                print("\n" + "=" * 90)
                print(f"RUN: {tag}")
                print("=" * 90)

                args_list = [
                    "--start", a.start,
                    "--end", a.end,
                    "--symbol", a.symbol,
                    "--exchange", a.exchange,
                    "--balance", str(a.balance),
                    "--daily-loss", str(a.daily_loss),
                    "--max-loss", str(a.max_loss),
                    "--fee", str(fee),
                    "--slip", str(slip),
                    "--sl", str(a.sl),
                    "--tp", str(tp),
                ]

                rc, stdout, stderr, cmd = run_one(a.python, a.script, args_list)

                # Toujours afficher la sortie du pipeline (copiable)
                print(stdout)
                if stderr.strip():
                    print("----- STDERR -----")
                    print(stderr)

                metrics = extract_final_metrics(stdout)
                if metrics is None:
                    print("!! Impossible de parser FINAL METRICS dans stdout. Run ignoré.")
                    continue

                # Copier results -> results_runs/tag
                safe_copytree(a.results_dir, outdir)

                # MDD additionnel
                mdd = compute_max_drawdown_from_equity_csv(os.path.join(outdir, "equity.csv"))
                if mdd:
                    metrics["mdd_abs"] = mdd["mdd_abs"]
                    metrics["mdd_pct"] = mdd["mdd_pct"]

                # ligne de synthèse courte
                line = {
                    "tp": tp,
                    "fee": fee,
                    "slip": slip,
                    "net_pnl": metrics.get("net_pnl"),
                    "profit_factor": metrics.get("profit_factor"),
                    "winrate": metrics.get("winrate"),
                    "n_trades": metrics.get("n_trades"),
                    "final_equity": metrics.get("final_equity"),
                    "mdd_abs": metrics.get("mdd_abs"),
                    "mdd_pct": metrics.get("mdd_pct"),
                    "failed": metrics.get("failed"),
                    "fail_reason": metrics.get("fail_reason"),
                    "run_dir": outdir,
                }
                summary.append(line)

                print("---- SUMMARY LINE (copiable) ----")
                print(json.dumps(line, indent=2, sort_keys=True))

    # Tri final par profit factor puis net pnl
    print("\n" + "#" * 90)
    print("TOP RUNS (profit_factor desc, net_pnl desc) — copiable")
    print("#" * 90)

    def keyfn(x):
        pf = x["profit_factor"]
        pnl = x["net_pnl"]
        pf = pf if isinstance(pf, (int, float)) else -1e9
        pnl = pnl if isinstance(pnl, (int, float)) else -1e9
        return (pf, pnl)

    top = sorted(summary, key=keyfn, reverse=True)[:10]
    print(json.dumps(top, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

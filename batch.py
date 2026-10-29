#!/usr/bin/env python3
"""
Run the solver on many DISPLIB instances, verify every solution with the official
verifier, and write one CSV row per instance.

AI-generated (Claude), October 2026.

Example (4 instances in parallel, 300 s each, prove optimality):
  python batch.py problems/ --pattern "nor1_*" --time 300 --gap 0 -j 4 \
      --verifier displib_verify.py --best solutions/ --out results.csv

Each instance runs in its own process (a crash or a license error only loses that row).
Extra solver flags can be passed after "--", e.g.  -- --heuristic --iter-limit 20
"""

import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))


def best_known(best_dir):
    out = {}
    if not best_dir:
        return out
    for f in glob.glob(os.path.join(best_dir, "*.json")):
        base = os.path.basename(f)[:-5]
        # solution files are named <instance>_<team>.json
        try:
            val = json.load(open(f))["objective_value"]
        except Exception:
            continue
        name = base.rsplit("_", 1)[0]
        out[name] = (val, base.rsplit("_", 1)[1])
    return out


def run_one(prob, args, extra):
    name = os.path.basename(prob)[:-5]
    sol = os.path.join(args.outdir, name + "_sol.json")
    stats = os.path.join(args.outdir, name + "_stats.json")
    cmd = [sys.executable, os.path.join(HERE, "run.py"), prob, "-o", sol, "--time", str(args.time),
           "--gap", str(args.gap), "--quiet", "--stats", stats]
    if args.threads:
        cmd += ["--threads", str(args.threads)]
    cmd += extra
    row = {"instance": name}
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=args.time * 2 + 120)
        out = p.stdout.strip().splitlines()
        try:
            res = json.loads(out[-1]) if out else {}
        except json.JSONDecodeError:
            res = {}
        if not res:
            row["status"] = "crash: " + (p.stderr.strip().splitlines() or ["?"])[-1][:200]
    except subprocess.TimeoutExpired:
        res = {}
        row["status"] = "killed (timeout)"
    except Exception as e:
        res = {}
        row["status"] = f"error: {e}"
    for k in ("status", "objective", "lower_bound", "time", "rounds", "conflicts",
              "agreement_cuts", "swap_cuts", "max_group", "groups"):
        if k in res:
            row[k] = res[k]
    if res.get("objective") is not None and res.get("lower_bound") is not None and res["objective"] > 0:
        row["gap"] = round((res["objective"] - res["lower_bound"]) / res["objective"], 6)
    if os.path.exists(stats):
        try:
            row["warnings"] = len(json.load(open(stats)).get("warnings", []))
        except Exception:
            pass
    if args.verifier and os.path.exists(sol) and res.get("objective") is not None:
        v = subprocess.run([sys.executable, args.verifier, prob, sol], capture_output=True, text=True)
        m = re.search(r"feasible with objective value (\d+)", v.stdout)
        row["verified"] = bool(m)
        row["verified_objective"] = int(m.group(1)) if m else None
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("problems", help="folder with DISPLIB problem .json files")
    ap.add_argument("--pattern", default="*", help="glob on the instance name, e.g. 'nor1_*'")
    ap.add_argument("--time", type=float, default=300.0)
    ap.add_argument("--gap", type=float, default=1e-4)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("-j", "--jobs", type=int, default=1)
    ap.add_argument("--verifier", default=None, help="path to displib_verify.py")
    ap.add_argument("--best", default=None, help="folder with best-known solutions (for comparison)")
    ap.add_argument("--outdir", default="batch_out")
    ap.add_argument("--out", default="results.csv")
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        k = argv.index("--")
        argv, extra = argv[:k], argv[k + 1:]
    args = ap.parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)
    probs = sorted(glob.glob(os.path.join(args.problems, args.pattern + ".json")))
    bk = best_known(args.best)
    cols = ["instance", "status", "objective", "lower_bound", "gap", "best_known", "best_team", "diff_to_best",
            "verified", "verified_objective", "time", "rounds", "conflicts", "agreement_cuts", "swap_cuts",
            "max_group", "groups", "warnings"]
    rows = []
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for row in ex.map(lambda pr: run_one(pr, args, extra), probs):
            b = bk.get(row["instance"])
            if b:
                row["best_known"], row["best_team"] = b
                if row.get("objective") is not None:
                    row["diff_to_best"] = row["objective"] - b[0]
            rows.append(row)
            print(", ".join(f"{k}={row.get(k)}" for k in cols if k in row), flush=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {args.out} ({len(rows)} instances)")


if __name__ == "__main__":
    main()

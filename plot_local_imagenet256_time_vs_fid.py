
from pathlib import Path
import json, re, csv, os

ROOT = Path(".")
JSON_ROOT = ROOT / "jsons"
CKPT_TSV = ROOT / "checkpoint_mtimes.tsv"

RUNS = {
    "Sinkhorn / OT": "imagenet256_ot_1node_4gpu_official30k",
    "Direct Riesz": "imagenet256_riesz_1node_4gpu_official30k",
    "Powered Riesz": "imagenet256_riesz_power_1node_4gpu_official30k",
    "Powered Riesz top-k": "imagenet256_riesz_power_topk_lr4_1node_4gpu_official30k",
}

TARGET_CFG = float(os.environ.get("CFG_SCALE", "1.5"))
CFG_TAG = str(TARGET_CFG).replace(".", "p")

OUT_CSV = Path(f"imagenet256_fid_vs_time_cfg{CFG_TAG}.csv")
OUT_PNG = Path(f"imagenet256_fid_vs_time_cfg{CFG_TAG}.png")
OUT_PDF = Path(f"imagenet256_fid_vs_time_cfg{CFG_TAG}.pdf")

def parse_step(path):
    m = re.search(r"state_(\d+)", str(path))
    return int(m.group(1)) if m else None

def read_ckpt_times():
    out = {}
    if not CKPT_TSV.exists():
        return out
    with open(CKPT_TSV) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p, t = line.split("\t")
            step = parse_step(p)
            if step is None:
                continue
            run = p.split("/")[0]
            out.setdefault(run, {})[step] = float(t)
    return out

def fit_line(step_to_time):
    items = sorted(step_to_time.items())
    if len(items) < 2:
        return None
    xs = [float(s) for s, _ in items]
    ys = [float(t) for _, t in items]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return None
    slope = num / den
    intercept = my - slope * mx
    return slope, intercept

def read_jsons_for_run(run):
    run_root = JSON_ROOT / run
    rows = []
    if not run_root.exists():
        return rows

    for p in sorted(run_root.rglob("*.json")):
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception:
            continue
        if "fid" not in d:
            continue

        cfg = d.get("cfg_scale", None)
        if cfg is not None:
            try:
                if abs(float(cfg) - TARGET_CFG) > 1e-8:
                    continue
            except Exception:
                pass

        step = d.get("step")
        if step is None:
            step = parse_step(p)
        if step is None:
            continue

        rows.append({
            "step": int(step),
            "fid": float(d["fid"]),
            "isc_mean": d.get("isc_mean", ""),
            "isc_std": d.get("isc_std", ""),
            "gen_time_eval_seconds": d.get("gen_time", ""),
            "json_path": str(p),
        })

    by_step = {}
    for r in rows:
        by_step[r["step"]] = r
    return [by_step[s] for s in sorted(by_step)]

ckpt_times_all = read_ckpt_times()
all_rows = []

for method, run in RUNS.items():
    rows = read_jsons_for_run(run)
    if not rows:
        print(f"[WARN] no rows for {run}")
        continue

    step_to_time = ckpt_times_all.get(run, {})
    fit = fit_line(step_to_time)

    if fit is not None:
        slope, intercept = fit
        for r in rows:
            step = r["step"]
            t = step_to_time.get(step, slope * step + intercept)
            r["elapsed_hours"] = (t - intercept) / 3600.0
            r["time_source"] = "checkpoint_mtime_fit"
    else:
        for r in rows:
            r["elapsed_hours"] = ""
            r["time_source"] = "missing_checkpoint_times"

    for r in rows:
        r["method"] = method
        r["run_name"] = run
        all_rows.append(r)

all_rows.sort(key=lambda r: (list(RUNS).index(r["method"]), r["step"]))

fields = [
    "method", "run_name", "step", "elapsed_hours", "fid",
    "isc_mean", "isc_std", "gen_time_eval_seconds", "time_source", "json_path",
]

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for r in all_rows:
        writer.writerow({k: r.get(k, "") for k in fields})

print(f"Saved CSV: {OUT_CSV}")
print()
print(f"{'method':24s} {'step':>8s} {'hours':>10s} {'fid':>10s}")
for r in all_rows:
    h = r["elapsed_hours"]
    h_str = f"{h:.2f}" if isinstance(h, float) else str(h)
    print(f"{r['method']:24s} {r['step']:8d} {h_str:>10s} {r['fid']:10.4f}")

try:
    import matplotlib.pyplot as plt
except Exception:
    print("\nmatplotlib not available, but CSV was created.")
    raise SystemExit(0)

plt.figure(figsize=(8.5, 5.5))
for method in RUNS:
    rows = [r for r in all_rows if r["method"] == method and isinstance(r["elapsed_hours"], float)]
    rows.sort(key=lambda r: r["elapsed_hours"])
    if rows:
        plt.plot([r["elapsed_hours"] for r in rows], [r["fid"] for r in rows], marker="o", label=method)

plt.xlabel("Training time, hours")
plt.ylabel("FID-50K")
plt.title(f"ImageNet256 FID vs training time, CFG {TARGET_CFG}")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200)
plt.savefig(OUT_PDF)
print(f"\nSaved PNG: {OUT_PNG}")
print(f"Saved PDF: {OUT_PDF}")

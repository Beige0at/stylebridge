"""Apples-to-apples run on the Mould & Rosin (NPRgeneral) content set.

Mirrors the Ioannou & Maddock (2024) Table-IX protocol: every content x style
pair (20 x 10 = 200) for each config, scored with SSIM / LPIPS / SIFID, then
averaged over the 200 pairs.

Content: nprgeneral/*.jpg
Style:   styles/*.jpg  (the paper doesn't list its 10 styles, so we use a
                        documented public-domain set and say so in the report)
Resolution 400. Resume-aware: pairs already in the CSV get skipped.
"""

import csv
import subprocess
import sys
import time
from pathlib import Path

from run_benchmarks import ENGINE, measure_lpips, measure_sifid, measure_ssim

ROOT = Path(__file__).resolve().parent
CONTENT_DIR = ROOT / "nprgeneral"
STYLE_DIR = ROOT / "styles"
MR_RESULTS = ROOT / "results" / "mould_rosin"
MR_RESULTS.mkdir(parents=True, exist_ok=True)
CSV_PATH = MR_RESULTS / "mould_rosin_benchmark.csv"

# config name -> extra engine CLI flags
CONFIGS = {
    "swd_adam": {
        "--optimizer": "adam", "--style-mode": "swd",
        "--resolution": 400, "--num-steps": 250, "--num-projections": 64,
    },
    "swd_lbfgs": {
        "--optimizer": "lbfgs", "--style-mode": "swd",
        "--resolution": 400, "--num-steps": 30,
    },
    "gram_adam": {
        "--optimizer": "adam", "--style-mode": "gram",
        "--resolution": 400, "--num-steps": 250,
    },
}


def load_done():
    # set of (config, content, style) already in the CSV
    done = set()
    if CSV_PATH.exists():
        with open(CSV_PATH) as f:
            for row in csv.DictReader(f):
                done.add((row["config"], row["content"], row["style"]))
    return done


def _valid_output(path):
    """A resumed output only counts if it exists and isn't empty.

    A zero-byte file left behind by a killed run would otherwise get scored
    as if it were a real result.
    """
    return path.exists() and path.stat().st_size > 0


def run_pair(content, style, config, config_args):
    out = MR_RESULTS / f"{config}__{content.stem}__{style.stem}.jpg"
    if not _valid_output(out):
        # flatten the config dict into --flag value pairs
        cmd = [sys.executable, str(ENGINE), str(content), str(style),
               "--out", str(out)] + [flag for k, v in config_args.items()
                                     for flag in (k, str(v))]
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"    !! {content.stem}/{style.stem} failed: "
                  f"{(proc.stderr or proc.stdout)[-200:]}")
            return None
        wall = time.time() - t0
    else:
        wall = float("nan")                 # reused an existing output
    row = {
        "ssim": measure_ssim(content, out),
        "lpips": measure_lpips(content, out),
        "sifid": measure_sifid(style, out),
        "wall_s": round(wall, 1),
    }
    print(f"    {content.stem:10s} x {style.stem:10s} -> "
          f"ssim={row['ssim']:.3f} lpips={row['lpips']:.3f} sifid={row['sifid']:.2f}")
    return row


def main():
    contents = sorted(CONTENT_DIR.glob("*.jpg"))
    styles = sorted(STYLE_DIR.glob("*.jpg"))
    assert len(contents) == 20, f"expected 20 content, got {len(contents)}"
    assert len(styles) == 10, f"expected 10 styles, got {len(styles)}"
    done = load_done()

    header = ["config", "content", "style", "ssim", "lpips", "sifid", "wall_s"]
    # append mode, so keep whatever previous runs already wrote
    f = open(CSV_PATH, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=header)
    if CSV_PATH.stat().st_size == 0:
        writer.writeheader()
    f.flush()

    for config, config_args in CONFIGS.items():
        agg = {"ssim": [], "lpips": [], "sifid": []}
        print(f"== config: {config} ==")
        for content in contents:
            for style in styles:
                if (config, content.stem, style.stem) in done:
                    continue                # already done in a previous run
                row = run_pair(content, style, config, config_args)
                if row is None:
                    continue                # this pair failed; try again next run
                writer.writerow({"config": config, "content": content.stem,
                                 "style": style.stem, **row})
                f.flush()                   # flush per pair so a crash loses little
                done.add((config, content.stem, style.stem))
                for k in agg:
                    agg[k].append(row[k])
        if agg["ssim"]:
            # average only the finite values so one bad pair can't poison the mean
            def _mean(vals):
                xs = [v for v in vals if v == v]          # drop NaN
                return sum(xs) / len(xs) if xs else float("nan")
            n = sum(1 for v in agg["ssim"] if v == v)
            print(f"[config {config}] n={n} "
                  f"mean SSIM={_mean(agg['ssim']):.4f} "
                  f"mean LPIPS={_mean(agg['lpips']):.4f} "
                  f"mean SIFID={_mean(agg['sifid']):.2f}")
    f.close()


if __name__ == "__main__":
    main()

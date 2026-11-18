#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Бенчмарк бэкендов RuCLIP (Фаза 5).

Запускает один и тот же exe с `--backend=torch|trt|dnn --repeat=N` и печатает
сравнение времени выполнения основной операции (zero-shot или Relevancy).

Пример:
  python tools/run_bench.py --exe ../rel_all/Release/RuCLIP.exe ^
      --clip ../CLIP/data/ruclip-vit-large-patch14-336 --repeat 10
"""

import argparse
import os
import re
import subprocess
import sys

TEST_IMAGES = ["1.png", "2.jpg", "3.jpg"]
BAG_IMAGES = ["bag2_1.jpg", "bag0_0.jpg", "bag1_0.jpg", "bag2_0.jpg", "bag2_2.jpg", "bag5_2.jpg"]
ZERO_SHOT_TEXT = "cat,bear,fox"
RELEVANCY_TEXT = "person,car,truck,bus"

PHASE_RE = re.compile(
    r"^\s+(\S+)\s+min\s+([\d.]+)\s+avg\s+([\d.]+)\s+max\s+([\d.]+)\s+ms"
)


def run_backend(exe, backend, clip, images, text, test_ind, repeat, env, extra_args=None):
    args = [
        exe,
        f"--backend={backend}",
        f"--test_ind={test_ind}",
        f"--imgs={','.join(images)}",
        f"--text={text}",
        f"--clip={clip}",
        f"--repeat={repeat}",
        "--show=0",
    ] + list(extra_args or [])
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )


def parse_timings(stdout):
    phases = {}
    for line in stdout.splitlines():
        m = PHASE_RE.match(line)
        if m:
            phases[m.group(1)] = float(m.group(3))  # avg
    return phases


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    here = os.path.dirname(os.path.abspath(__file__))
    default_exe = os.path.normpath(os.path.join(here, "..", "..", "rel_all", "Release", "RuCLIP.exe"))
    default_clip = os.path.normpath(os.path.join(here, "..", "..", "CLIP", "data",
                                                 "ruclip-vit-large-patch14-336"))

    p = argparse.ArgumentParser(description="Benchmark RuCLIP backends")
    p.add_argument("--exe", default=default_exe)
    p.add_argument("--clip", default=default_clip)
    p.add_argument("--backends", default="torch,trt,dnn")
    p.add_argument("--test-ind", type=int, default=0, choices=[0, 1],
                   help="0 — zero-shot, 1 — Relevancy")
    p.add_argument("--repeat", type=int, default=10)
    p.add_argument("--images-dir", default=None)
    p.add_argument("--extra-path", default="")
    args = p.parse_args()

    exe = os.path.abspath(args.exe)
    clip = os.path.abspath(args.clip)
    images_dir = args.images_dir or os.path.normpath(os.path.join(clip, "..", "test_images"))

    if not os.path.isfile(exe):
        sys.exit(f"exe не найден: {exe}")

    env = os.environ.copy()
    if args.extra_path:
        env["PATH"] = args.extra_path + os.pathsep + env.get("PATH", "")

    if args.test_ind == 0:
        images = [os.path.join(images_dir, n) for n in TEST_IMAGES]
        text = ZERO_SHOT_TEXT
        title = "zero-shot"
    else:
        images = [os.path.join(images_dir, n) for n in BAG_IMAGES]
        text = RELEVANCY_TEXT
        title = "Relevancy"

    specs = [b.strip() for b in args.backends.split(",") if b.strip()]
    results = {}
    order = []

    for spec in specs:
        # "trt:fp16" -> backend=trt, label=trt_fp16, доп. аргумент --fp16=1.
        if ":" in spec:
            backend, opt = spec.split(":", 1)
            extra = ["--fp16=1"] if opt == "fp16" else []
            label = f"{backend}_{opt}"
        else:
            backend, extra, label = spec, [], spec

        p = run_backend(exe, backend, clip, images, text, args.test_ind, args.repeat, env, extra)
        if p.returncode != 0 and "not built" in (p.stderr + p.stdout):
            print(f"[skip] {label}: не собран в этом exe")
            continue
        if p.returncode != 0:
            print(f"[FAIL] {label}: exit={p.returncode}\n{(p.stderr or p.stdout)[-800:]}")
            return 1

        timings = parse_timings(p.stdout)
        if "total" not in timings:
            print(f"[FAIL] {label}: в выводе нет Timings (--repeat задан?)")
            return 1
        results[label] = timings
        if not order:
            order = [k for k in timings if k != "total"] + ["total"]
        print(f"[ok]   {label}: total avg = {timings['total']:.2f} ms")

    if not results:
        print("Ни один бэкенд не запустился.")
        return 1

    reference = "torch" if "torch" in results else next(iter(results))
    base_total = results[reference]["total"]

    print(f"\n{title}, repeats={args.repeat}, эталон: {reference}\n")

    header = f"{'backend':8s} {reference:>10s} {'total, ms':>10s} {'vs ' + reference:>14s}"
    print(header)
    print("-" * len(header))
    for backend, t in results.items():
        rel = base_total / t["total"] if t["total"] > 0 else float("nan")
        print(f"{backend:8s} {'':>10s} {t['total']:10.2f} {rel:13.2f}x")

    print("\nПо фазам, avg ms:")
    cols = [c for c in order if c != "total"]
    print(f"{'backend':8s} " + " ".join(f"{c:>13s}" for c in cols) + f" {'total':>13s}")
    for backend, t in results.items():
        row = " ".join(f"{t.get(c, float('nan')):13.2f}" for c in cols)
        print(f"{backend:8s} {row} {t['total']:13.2f}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

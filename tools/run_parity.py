#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Паритет-тест бэкендов RuCLIP (Фаза 5).

Запускает один и тот же exe с разными --backend и сравнивает:
  * матрицу вероятностей zero-shot (test_ind=0);
  * значения Relevancy (test_ind=1).

Пример (PATH должен содержать OpenCV/TensorRT/CUDA, как для run_all.bat):
  python tools/run_parity.py --exe ../rel_all/Release/RuCLIP.exe ^
      --clip ../CLIP/data/ruclip-vit-large-patch14-336

Код возврата: 0 — паритет в допуске, 1 — расхождение/ошибка.
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

NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def run_backend(exe, backend, clip, images, text, test_ind, env, extra_args=None):
    args = [
        exe,
        f"--backend={backend}",
        f"--test_ind={test_ind}",
        f"--imgs={','.join(images)}",
        f"--text={text}",
        f"--clip={clip}",
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


def parse_zero_shot(stdout):
    lines = stdout.splitlines()
    start = None
    for i, line in enumerate(lines):
        if "probs per image" in line:
            start = i + 1
            break
    if start is None:
        return None

    rows = []
    for line in lines[start:]:
        s = line.strip()
        if not s:
            if rows:
                break
            continue
        nums = NUM_RE.findall(s)
        if len(nums) < 2:
            break
        rows.append([float(x) for x in nums])
    return rows or None


def parse_relevancy(stdout):
    return [float(m.group(1)) for m in re.finditer(r"P\(image match\) = (\S+)", stdout)]


def matrix_diff(a, b):
    if a is None or b is None:
        return None
    if len(a) != len(b):
        return float("inf")
    d = 0.0
    for ra, rb in zip(a, b):
        if len(ra) != len(rb):
            return float("inf")
        for x, y in zip(ra, rb):
            d = max(d, abs(x - y))
    return d


def vec_diff(a, b):
    if a is None or b is None:
        return None
    if len(a) != len(b):
        return float("inf")
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    here = os.path.dirname(os.path.abspath(__file__))
    default_exe = os.path.normpath(os.path.join(here, "..", "..", "rel_all", "Release", "RuCLIP.exe"))
    default_clip = os.path.normpath(os.path.join(here, "..", "..", "CLIP", "data",
                                                 "ruclip-vit-large-patch14-336"))

    p = argparse.ArgumentParser(description="Parity test for RuCLIP backends")
    p.add_argument("--exe", default=default_exe)
    p.add_argument("--clip", default=default_clip)
    p.add_argument("--backends", default="torch,trt,dnn")
    p.add_argument("--images-dir", default=None,
                   help="каталог test_images (по умолчанию <clip>/../test_images)")
    p.add_argument("--tol", type=float, default=2e-3,
                   help="допуск max|diff| между бэкендами (TF32 в TRT ~5e-4)")
    p.add_argument("--tol-fp16", type=float, default=5e-3,
                   help="допуск для FP16-бэкендов (trt:fp16 даёт ~2.6e-3)")
    p.add_argument("--extra-path", default="",
                   help="дополнительный каталог(ы) для PATH через ';'")
    args = p.parse_args()

    exe = os.path.abspath(args.exe)
    clip = os.path.abspath(args.clip)
    images_dir = args.images_dir or os.path.normpath(os.path.join(clip, "..", "test_images"))

    if not os.path.isfile(exe):
        sys.exit(f"exe не найден: {exe}")
    if not os.path.isdir(clip):
        sys.exit(f"каталог модели не найден: {clip}")

    env = os.environ.copy()
    if args.extra_path:
        env["PATH"] = args.extra_path + os.pathsep + env.get("PATH", "")

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    test_imgs = [os.path.join(images_dir, n) for n in TEST_IMAGES]
    bag_imgs = [os.path.join(images_dir, n) for n in BAG_IMAGES]

    results_zero = {}
    results_rel = {}
    available = []

    for spec in backends:
        # "trt:fp16" -> backend=trt, label=trt_fp16, доп. аргумент --fp16=1.
        if ":" in spec:
            backend, opt = spec.split(":", 1)
            extra = ["--fp16=1"] if opt == "fp16" else []
            label = f"{backend}_{opt}"
        else:
            backend, extra, label = spec, [], spec

        p_zero = run_backend(exe, backend, clip, test_imgs, ZERO_SHOT_TEXT, 0, env, extra)
        if p_zero.returncode != 0 and "not built" in (p_zero.stderr + p_zero.stdout):
            print(f"[skip] {label}: не собран в этом exe")
            continue
        if p_zero.returncode != 0:
            print(f"[FAIL] {label}: exit={p_zero.returncode}\n{(p_zero.stderr or p_zero.stdout)[-800:]}")
            return 1

        rows = parse_zero_shot(p_zero.stdout)
        if rows is None:
            print(f"[FAIL] {label}: не удалось разобрать zero-shot вывод")
            return 1
        results_zero[label] = rows
        available.append(label)
        print(f"[ok]   {label}: zero-shot {len(rows)}x{len(rows[0])}")

        p_rel = run_backend(exe, backend, clip, bag_imgs, RELEVANCY_TEXT, 1, env, extra)
        if p_rel.returncode != 0:
            print(f"[FAIL] {label}: Relevancy exit={p_rel.returncode}\n{(p_rel.stderr or p_rel.stdout)[-800:]}")
            return 1
        rel = parse_relevancy(p_rel.stdout)
        if not rel:
            print(f"[FAIL] {label}: не удалось разобрать Relevancy вывод")
            return 1
        results_rel[label] = rel

    if len(available) < 2:
        print(f"\nДоступен только один бэкенд ({available}); сравнивать нечего — пропуск.")
        return 0

    reference = "torch" if "torch" in available else available[0]
    print(f"\nЭталон: {reference}\n")

    ok = True
    for backend in available:
        if backend == reference:
            continue
        tol = args.tol_fp16 if "fp16" in backend else args.tol
        d_zero = matrix_diff(results_zero[reference], results_zero[backend])
        d_rel = vec_diff(results_rel[reference], results_rel[backend])
        status_zero = "OK" if (d_zero is not None and d_zero <= tol) else "FAIL"
        status_rel = "OK" if (d_rel is not None and d_rel <= tol) else "FAIL"
        ok = ok and status_zero == "OK" and status_rel == "OK"
        print(f"{backend:9s} vs {reference:6s}: "
              f"zero-shot max|diff|={d_zero:.3e} [{status_zero}]  "
              f"relevancy max|diff|={d_rel:.3e} [{status_rel}]  (tol={tol:g})")

    print("\nPASS" if ok else f"\nFAIL (tol={args.tol:g}, fp16={args.tol_fp16:g})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Экспорт ruCLIP в ONNX (Фаза 2).

Экспортирует из оригинальной Python-модели ruclip две независимые модели:

  clip_visual.onnx   : pixel_values [B, 3, H, W] float32 -> image_features [B, D]
  clip_textual.onnx  : input_ids    [B, L]       int32   -> text_features  [B, L, D]

и дописывает logit_scale в config.json (для бэкендов TensorRT / OpenCV DNN).

Особенности:
  * только локальные файлы (config.json, pytorch_model.bin), без сети;
  * text-энкодер возвращает ПОЛНУЮ последовательность [B, L, D] — выбор EOS
    делается в C++ (GatherEos), чтобы не тащить динамический Gather в граф;
  * int32-индексы на входе text-энкодера (дружелюбно к OpenCV DNN);
  * исправления для динамического batch: flatten(2) вместо reshape и
    zeros_like вместо zeros (иначе batch запекается как константа).

Пример:
  python tools/export_onnx.py --models ruclip-vit-large-patch14-336
  python tools/export_onnx.py            # все три модели из ../CLIP/data
"""

import argparse
import importlib.util
import json
import os
import sys

import torch
import torch.nn as nn

DEFAULT_MODELS = [
    "ruclip-vit-large-patch14-336",
    "ruclip-vit-base-patch16-384",
    "ruclip-vit-base-patch32-384",
]


# ---------------------------------------------------------------------------
# Загрузка класса CLIP из vendored-репозитория ru-clip (без import ruclip,
# чтобы не тянуть huggingface_hub/youtokentome).
# ---------------------------------------------------------------------------
def load_clip_class(model_py_path):
    spec = importlib.util.spec_from_file_location("ruclip_model_export", model_py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load model.py: {model_py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CLIP


# ---------------------------------------------------------------------------
# Обёртки для экспорта
# ---------------------------------------------------------------------------
class ImageEncoder(nn.Module):
    """ViT-энкодер. Повторяет VisionTransformer.forward с двумя правками,
    необходимыми для корректного dynamic batch в ONNX:
      * x.flatten(2)            вместо reshape(x.shape[0], x.shape[1], -1);
      * torch.zeros_like(...)   вместо torch.zeros(x.shape[0], ...).
    """

    def __init__(self, clip):
        super().__init__()
        self.visual = clip.visual

    def forward(self, pixel_values):
        v = self.visual
        x = v.conv1(pixel_values)          # [B, width, grid, grid]
        x = x.flatten(2)                   # [B, width, grid*grid]
        x = x.permute(0, 2, 1)             # [B, grid*grid, width]
        cls = v.class_embedding.to(x.dtype) + torch.zeros_like(x[:, :1, :])
        x = torch.cat([cls, x], dim=1)     # [B, grid*grid + 1, width]
        x = x + v.positional_embedding.to(x.dtype)
        x = v.ln_pre(x)
        x = x.permute(1, 0, 2)             # NLD -> LND
        x = v.transformer(x)
        x = x.permute(1, 0, 2)             # LND -> NLD
        x = v.ln_post(x[:, 0, :])          # CLS-токен
        if v.proj is not None:
            x = x @ v.proj
        return x                           # [B, D]


class TextEncoder(nn.Module):
    """Текстовый энкодер: token_embedding -> transformer -> ln_final ->
    text_projection. Возвращает полную последовательность [B, L, D]
    (без выбора EOS-токена — это делает C++ через GatherEos).
    """

    def __init__(self, clip):
        super().__init__()
        self.token_embedding = clip.token_embedding
        self.positional_embedding = clip.positional_embedding
        self.transformer = clip.transformer
        self.ln_final = clip.ln_final
        self.text_projection = clip.text_projection

    def forward(self, input_ids):
        x = self.token_embedding(input_ids).type(torch.float32)  # [B, L, W]
        x = x + self.positional_embedding.type(torch.float32)
        x = x.permute(1, 0, 2)             # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)             # LND -> NLD
        x = self.ln_final(x).type(torch.float32)
        return x @ self.text_projection    # [B, L, D]


# ---------------------------------------------------------------------------
# Экспорт и проверка
# ---------------------------------------------------------------------------
def export_one(module, dummy, out_path, input_names, output_names, dynamic_axes, opset):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.onnx.export(
        module,
        dummy,
        out_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False,
        verbose=False,
    )


def verify_onnx(module, dummy, out_path, input_name, label):
    """Сравнивает выход PyTorch и ONNX Runtime на одном входе."""
    try:
        import onnxruntime as ort
    except Exception as e:
        print(f"  [verify] onnxruntime недоступен, пропуск: {e}")
        return

    import numpy as np

    with torch.no_grad():
        ref = module(dummy).detach().cpu().numpy()

    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    got = sess.run(None, {input_name: dummy.detach().cpu().numpy()})[0]

    if ref.shape != got.shape:
        print(f"  [verify] {label}: shape mismatch ref={ref.shape} got={got.shape}")
        return
    diff = float(np.max(np.abs(ref - got)))
    status = "OK" if diff < 1e-4 else "WARN"
    print(f"  [verify] {label}: max|diff| = {diff:.3e}  [{status}]  shape={ref.shape}")


def summary_io(out_path):
    try:
        import onnx
    except Exception:
        return
    m = onnx.load(out_path)
    defs = [(i.name, i.type.tensor_type.elem_type, [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim])
            for i in m.graph.input]
    outs = [(o.name, o.type.tensor_type.elem_type, [d.dim_param or d.dim_value for d in o.type.tensor_type.shape.dim])
            for o in m.graph.output]
    print(f"  inputs : {defs}")
    print(f"  outputs: {outs}")
    from collections import Counter
    ops = Counter(n.op_type for n in m.graph.node)
    print(f"  ops    : {dict(sorted(ops.items()))}")


def export_model(model_dir, clip_cls, opset, dynamic_batch, do_verify):
    print(f"\n=== {model_dir} ===")
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    clip = clip_cls.from_pretrained(model_dir).eval()
    image_size = int(config["image_resolution"])
    context_length = int(config["context_length"])

    image_enc = ImageEncoder(clip).eval()
    text_enc = TextEncoder(clip).eval()

    dynamic_axes_img = {"pixel_values": {0: "batch"}, "image_features": {0: "batch"}} if dynamic_batch else None
    dynamic_axes_txt = {"input_ids": {0: "batch"}, "text_features": {0: "batch"}} if dynamic_batch else None

    visual_path = os.path.join(model_dir, "clip_visual.onnx")
    textual_path = os.path.join(model_dir, "clip_textual.onnx")

    dummy_img = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)
    dummy_txt = torch.zeros(1, context_length, dtype=torch.int32)

    print("  exporting clip_visual.onnx ...")
    export_one(image_enc, dummy_img, visual_path, ["pixel_values"], ["image_features"],
               dynamic_axes_img, opset)
    print("  exporting clip_textual.onnx ...")
    export_one(text_enc, dummy_txt, textual_path, ["input_ids"], ["text_features"],
               dynamic_axes_txt, opset)

    summary_io(visual_path)
    summary_io(textual_path)

    if do_verify:
        verify_onnx(image_enc, dummy_img, visual_path, "pixel_values", "visual batch=1")
        verify_onnx(text_enc, dummy_txt, textual_path, "input_ids", "textual batch=1")
        if dynamic_batch:
            dummy_img2 = torch.randn(3, 3, image_size, image_size, dtype=torch.float32)
            dummy_txt2 = torch.zeros(3, context_length, dtype=torch.int32)
            verify_onnx(image_enc, dummy_img2, visual_path, "pixel_values", "visual batch=3")
            verify_onnx(text_enc, dummy_txt2, textual_path, "input_ids", "textual batch=3")

    # logit_scale: эффективный множитель exp(logit_scale), читается бэкендами.
    with torch.no_grad():
        logit_scale = float(clip.logit_scale.exp().detach().cpu())
    config["logit_scale"] = logit_scale
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"  config.json: logit_scale = {logit_scale:.6f}")

    return visual_path, textual_path


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    here = os.path.dirname(os.path.abspath(__file__))
    default_data_root = os.path.normpath(os.path.join(here, "..", "..", "CLIP", "data"))
    default_model_py = os.path.normpath(os.path.join(
        here, "..", "..", "CLIP", "docs", "ru-clip-main", "ruclip", "model.py"))

    p = argparse.ArgumentParser(description="Export ruCLIP encoders to ONNX")
    p.add_argument("--data-root", default=default_data_root,
                   help="каталог с папками моделей (по умолчанию: CLIP/data)")
    p.add_argument("--models", nargs="*", default=DEFAULT_MODELS,
                   help="имена моделей внутри --data-root")
    p.add_argument("--model-dir", nargs="*", default=None,
                   help="явные пути к каталогам моделей (перекрывает --data-root/--models)")
    p.add_argument("--model-py", default=default_model_py,
                   help="путь к ruclip/model.py (vendored)")
    p.add_argument("--opset", type=int, default=14)
    p.add_argument("--no-dynamic-batch", action="store_true",
                   help="экспортировать с фиксированным batch=1")
    p.add_argument("--no-verify", action="store_true", help="не проверять через onnxruntime")
    args = p.parse_args()

    if not os.path.isfile(args.model_py):
        sys.exit(f"model.py не найден: {args.model_py}")
    clip_cls = load_clip_class(args.model_py)

    if args.model_dir:
        model_dirs = [os.path.abspath(d) for d in args.model_dir]
    else:
        model_dirs = [os.path.join(args.data_root, m) for m in args.models]

    for d in model_dirs:
        if not os.path.isdir(d):
            sys.exit(f"каталог модели не найден: {d}")
        export_model(d, clip_cls, args.opset,
                     dynamic_batch=not args.no_dynamic_batch,
                     do_verify=not args.no_verify)

    print("\nГотово.")


if __name__ == "__main__":
    main()

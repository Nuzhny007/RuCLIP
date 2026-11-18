# RuCLIP (C++)

Неофициальный C++-порт [ruCLIP](https://github.com/ai-forever/ru-clip) (Sber AI) с **тремя
взаимозаменяемыми бэкендами инференса**: LibTorch, TensorRT и OpenCV DNN.

Один и тот же код собирается с любым набором бэкендов; нужный выбирается в рантайме:

```
RuCLIP.exe --backend=torch|trt|dnn ...
```

Токенизация (YouTokenToMe), препроцессинг изображений (OpenCV) и постобработка
(L2-норма, `logit_scale`, softmax, `Relevancy`) общие для всех бэкендов, поэтому
результаты совпадают с точностью до численных эффектов конкретного рантайма.

Оригинальная PyTorch-реализация: <https://github.com/ai-forever/ru-clip> ·
Оригинальная статья CLIP: <https://arxiv.org/pdf/2103.00020.pdf>

- Дизайн многобэкендной архитектуры: [`docs/backend-architecture.md`](docs/backend-architecture.md)
- Разбор кода: [`docs/architecture.md`](docs/architecture.md) · план улучшений: [`docs/roadmap.md`](docs/roadmap.md)

## Статус

Все фазы (0–5) завершены:

| Фаза | Что сделано |
|---|---|
| 0 | Нейтральный слой `src/Backend/`: `TensorF32`, `TextBatch`, `ImageBatch`, `ClipMath` (`Logits`/`Relevancy`/`GatherEos`) |
| 1 | `IClipBackend` + `TorchBackend`, ключ `--backend`, CMake-опции `RUCLIP_WITH_*` |
| 2 | Экспорт ONNX (`tools/export_onnx.py`) + `logit_scale` в `config.json` для трёх моделей |
| 3 | `TensorRTBackend` (engine из ONNX, кэш в `<model>/engines/`) |
| 4 | `DnnBackend` (OpenCV 5.x, новый graph-движок `ENGINE_OPENCV`) |
| 5 | Паритет-тест `tools/run_parity.py`, скрипты запуска, этот README |

## Возможности

- **Три бэкенда**: LibTorch (CPU/CUDA), TensorRT (NVIDIA GPU), OpenCV DNN (CPU, OpenCV 5.x).
- **Гибкая сборка**: один бэкенд или несколько в одном бинарнике (`RUCLIP_WITH_TORCH/TRT/DNN`).
- **Выбор в рантайме**: `--backend=torch|trt|dnn`; если бэкенд не собран — понятная ошибка.
- **Единый контракт**: `EncodeImage([B,3,H,W]) -> [B,D]`, `EncodeText([B,77]) -> [B,D]`
  (int32-токены, эмбеддинг по позиции EOS), постобработка общая.
- **ONNX-экспорт** из оригинальной Python-модели (без сети, только локальные веса).
- **Паритет-тест** бэкендов на эталонных входах.

## Требования

| Компонент | Версия / примечание |
|---|---|
| CMake | ≥ 3.18, C++17 |
| OpenCV | ≥ 4.6 (I/O); **≥ 5.x — для DNN-бэкенда** |
| LibTorch | `-DRUCLIP_WITH_TORCH=ON`; нужен `jit_model.zip` в каталоге модели |
| TensorRT + CUDA | `-DRUCLIP_WITH_TRT=ON`; проверено на TensorRT 10.16.1.11 + CUDA 13.2, GPU sm89 |
| YouTokenToMe, nlohmann/json | vendored в `src/` |

## Модели

| Модель | embed_dim | image_res | vision | text | `jit_model.zip` | ONNX |
|---|---|---|---|---|---|---|
| `ruclip-vit-large-patch14-336` | 768 | 336 | L=24, W=1024, p=14 | L=12, W=768, h=12 | ✅ | ✅ |
| `ruclip-vit-base-patch16-384` | 512 | 384 | L=12, W=768, p=16 | L=12, W=512, h=8 | ❌ | ✅ |
| `ruclip-vit-base-patch32-384` | 512 | 384 | L=12, W=768, p=32 | L=12, W=512, h=8 | ❌ | ✅ |

> Torch-бэкенд требует `jit_model.zip` (TorchScript). Для base-моделей его можно
> экспортировать из Python (см. приложение) или использовать `trt`/`dnn`.

## Сборка

Опции CMake: `RUCLIP_WITH_TORCH` (ON), `RUCLIP_WITH_TRT` (OFF), `RUCLIP_WITH_DNN` (OFF).
Нужно включить хотя бы один бэкенд.

**Torch-only:**
```bat
cmake -S RuCLIP -B rel -G "Visual Studio 17 2022" -A x64 -DRUCLIP_WITH_TORCH=ON
cmake --build rel --config Release --target RuCLIP
```

**TensorRT-only** (пути — пример для этой машины):
```bat
cmake -S RuCLIP -B rel_trt -G "Visual Studio 17 2022" -A x64 ^
  -DRUCLIP_WITH_TORCH=OFF -DRUCLIP_WITH_TRT=ON ^
  "-DRUCLIP_TRT_ROOT=C:/work/libraries/cuda/TensorRT-10.16.1.11" ^
  "-DCUDAToolkit_ROOT=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2"
cmake --build rel_trt --config Release
```

**OpenCV DNN-only** (нужен OpenCV 5.x):
```bat
cmake -S RuCLIP -B rel_dnn -G "Visual Studio 17 2022" -A x64 ^
  -DRUCLIP_WITH_TORCH=OFF -DRUCLIP_WITH_DNN=ON ^
  "-DOpenCV_DIR=C:/work/libraries/opencv/5.0/build_cuda/install/x64/vc17/lib"
cmake --build rel_dnn --config Release
```

**Все три бэкенда в одном exe:**
```bat
cmake -S RuCLIP -B rel_all -G "Visual Studio 17 2022" -A x64 ^
  -DRUCLIP_WITH_TORCH=ON -DRUCLIP_WITH_TRT=ON -DRUCLIP_WITH_DNN=ON ^
  "-DTorch_DIR=C:/work/libraries/torch/libtorch-win-shared-with-deps-latest+cu126/libtorch/share/cmake/Torch" ^
  "-DOpenCV_DIR=C:/work/libraries/opencv/5.0/build_cuda/install/x64/vc17/lib" ^
  "-DRUCLIP_TRT_ROOT=C:/work/libraries/cuda/TensorRT-10.16.1.11" ^
  "-DCUDAToolkit_ROOT=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2"
cmake --build rel_all --config Release
```

> Если CMake не находит зависимости, укажите пути явно: `-DOpenCV_DIR=...` и
> `-DTorch_DIR=...`. `RUCLIP_TRT_ROOT` при необходимости определяется автоматически из
> `C:/work/libraries/cuda/TensorRT-*` или переменной окружения `TENSORRT_ROOT`.

## Экспорт ONNX

Скрипт `tools/export_onnx.py` экспортирует из **оригинальной Python-модели** `ruclip`
(офлайн, только локальные `config.json`/`pytorch_model.bin`):

- `clip_visual.onnx` — `pixel_values [B,3,H,W] float32 -> image_features [B,D]`;
- `clip_textual.onnx` — `input_ids [B,77] int32 -> text_features [B,77,D]`
  (полная последовательность; EOS выбирается в C++ через `GatherEos`);
- `logit_scale = exp(logit_scale)` дописывается в `config.json`.

```bat
python tools/export_onnx.py                       :: все три модели из ../CLIP/data
python tools/export_onnx.py --models ruclip-vit-large-patch14-336
```

Требуется Python + PyTorch + onnx (для проверки — onnxruntime). Особенности: opset 14,
dynamic batch, разложенные `LayerNorm`/`QuickGELU`. `logit_scale` **обучаемый и разный по
моделям** (31.40 / 45.29 / 48.74), поэтому читается из `config.json`, а не хардкодится.

## Запуск

```
RuCLIP.exe --test_ind=0|1 --imgs=<img1,img2,...> --text=<label1,label2,...> ^
           --clip=<model_dir> [--backend=torch|trt|dnn] [--preprocess=letterbox|centercrop] ^
           [--show=0|1] [--max_batch=8] [--fp16=0|1]
```

| Ключ | Значение |
|---|---|
| `--test_ind` | `0` — изображения × тексты (zero-shot); `1` — первое изображение против остальных (`Relevancy`) |
| `--backend` | `torch` (по умолчанию), `trt`, `dnn` |
| `--preprocess` | `letterbox` (по умолчанию) или `centercrop` |
| `--max_batch` | верхняя граница batch для optimization profile TensorRT |
| `--fp16` | TensorRT: построить FP16-engine |
| `--repeat` | число измеряемых итераций основной операции (по умолчанию 1); прогрев не учитывается |
| `--show` | `0` — без окон OpenCV |

**Zero-shot (три изображения × метки):**
```bat
set IMGS=..\CLIP\data\test_images\1.png,..\CLIP\data\test_images\2.jpg,..\CLIP\data\test_images\3.jpg
set CLIP=..\CLIP\data\ruclip-vit-large-patch14-336
RuCLIP.exe --backend=trt --test_ind=0 --imgs=%IMGS% --text=cat,bear,fox --clip=%CLIP% --show=0
```

**Поиск похожих изображений:**
```bat
set IMGS=..\CLIP\data\test_images\bag2_1.jpg,..\CLIP\data\test_images\bag0_0.jpg,..\CLIP\data\test_images\bag1_0.jpg
RuCLIP.exe --backend=dnn --test_ind=1 --imgs=%IMGS% --text=person,car,truck,bus --clip=%CLIP% --show=0
```

**PATH для рантайма:**
- `torch`: каталог `rel\Release` (DLL LibTorch) + `bin` OpenCV.
- `trt`: `TensorRT\bin`, `CUDA\v13.2\bin\x64`, `bin` OpenCV, каталог exe (DLL cuDNN).
- `dnn`: `OpenCV5\install\x64\vc17\bin`, `CUDA\v13.2\bin\x64`, `cudnn64_9.dll`, `onnxruntime.dll`.

Готовые скрипты: `rel_all\Release\run_all.bat torch|trt|dnn`, `rel_multi\Release\run_trt.bat`.

## Паритет-тест

`tools/run_parity.py` прогоняет один и тот же exe с разными `--backend` и сравнивает
матрицу zero-shot и значения `Relevancy`:

```bat
python tools\run_parity.py --exe ..\rel_all\Release\RuCLIP.exe ^
    --clip ..\CLIP\data\ruclip-vit-large-patch14-336
```

Пример результата (эталон — torch):
```
trt vs torch : zero-shot max|diff|=4.8e-04 [OK]  relevancy max|diff|=3.9e-05 [OK]
dnn vs torch : zero-shot max|diff|=5.0e-06 [OK]  relevancy max|diff|=4.7e-05 [OK]
PASS
```

Допуск по умолчанию `2e-3` (TF32 в TensorRT даёт ~5e-4). Скрипт возвращает 0/1 —
его можно использовать в CI.

## Замеры производительности

`--repeat=N` выполняет основную операцию N раз (после одного прогрева) и печатает
`min/avg/max` по фазам: `preprocess`, `encode_text`, `encode_image`, `postprocess`, `total`.

Сравнить бэкенды (включая FP16 для TensorRT) одним запуском:

```bat
python tools\run_bench.py --exe ..\rel_all\Release\RuCLIP.exe ^
    --clip ..\CLIP\data\ruclip-vit-large-patch14-336 --repeat 10 ^
    --backends torch,trt,trt:fp16,dnn
```

Формат `trt:fp16` означает `--backend=trt --fp16=1`; для `Relevancy` добавьте
`--test-ind 1` (фазы: `reference`, `others`, `text`, `relevancy`).

Пример (large-336, 3 изображения × 3 метки, RTX 4070; `rel_all`):

```
backend       torch  total, ms       vs torch
---------------------------------------------
torch                   180.14          1.00x
trt                     104.97          1.72x
trt_fp16                 34.70          5.19x
dnn                    1934.95          0.09x

По фазам, avg ms:
backend     preprocess   encode_text  encode_image   postprocess         total
torch             4.24          6.84        169.04          0.02        180.14
trt               4.47          4.60         95.88          0.03        104.97
trt_fp16          4.24          4.82         25.62          0.02         34.70
dnn               4.71         92.11       1838.09          0.04       1934.95
```

- **FP16 (TensorRT)** — ~5× к torch и ~3× к FP32-движку TensorRT; основной выигрыш в `encode_image`.
- **Точность FP16**: `max|diff|` вероятностей относительно torch ≈ `2.6e-3` (у FP32-TRT
  `4.8e-4`, у DNN `5e-6`); паритет-тест использует для FP16 отдельный допуск
  `--tol-fp16 5e-3` (по умолчанию).
- `--fp16` поддерживается **только TensorRT**; `torch` и `dnn` его игнорируют.
- Время сборки engine и загрузки моделей/графов в замер не входит (конструктор бэкенда).
  DNN в OpenCV 5.1.0 сейчас CPU-only. Абсолютные числа зависят от модели, GPU и текущей
  загрузки — сравнивайте на своей машине.

## Структура проекта

```
RuCLIP/
├── CMakeLists.txt              # опции бэкендов, поиск Torch/TensorRT/OpenCV
├── CMakeLists.Files.txt        # условные списки исходников
├── src/
│   ├── Backend/
│   │   ├── ClipTypes.h         # TensorF32 / TextBatch / ImageBatch / BackendConfig
│   │   ├── ClipMath.*          # Logits / ProbsPerImage / Relevancy / GatherEos
│   │   ├── IClipBackend.h      # интерфейс + CreateBackend
│   │   ├── TorchBackend.*
│   │   ├── TensorRTBackend.*
│   │   ├── DnnBackend.*
│   │   └── BackendFactory.cpp
│   ├── RuCLIP.*                # LibTorch-модель (CLIPImpl)
│   ├── RuCLIPProcessor.*       # токенизация + препроцессинг (нейтральные буферы)
│   ├── main.cpp                # CLI
│   └── youtokentome/           # vendored BPE
├── tools/
│   ├── export_onnx.py          # экспорт clip_visual/clip_textual + logit_scale
│   ├── run_parity.py           # паритет-тест бэкендов
│   └── run_bench.py            # замеры производительности бэкендов
├── tests/                      # unit-тест ClipMath
└── docs/                       # архитектура, roadmap, дизайн бэкендов
```

## Известные ограничения и расхождения

- **TensorRT** — только NVIDIA GPU; engine привязан к GPU/версии TensorRT (пересобирается
  автоматически в `engines/`); по умолчанию TF32 (для строгого FP32 — `clearFlag(kTF32)`).
- **OpenCV DNN** — требует OpenCV 5.x; новый движок `ENGINE_OPENCV` сейчас CPU-only;
  `opencv_dnn` собран с CUDA, поэтому в PATH нужны cuDNN/ONNX Runtime.
- **LibTorch** — требует `jit_model.zip`; для base-моделей его надо экспортировать.
- **Препроцессинг**: C++ читает изображения через `cv::imread` в **BGR** и не конвертирует
  в RGB, тогда как Python-референс использует RGB. Это даёт систематическое расхождение с
  Python-вероятностями; исправление — `cv::cvtColor(img, img, cv::COLOR_BGR2RGB)` перед
  нормализацией (отдельная задача). Числа `0.8879 / 0.0063 / 0.1058` ниже — из Python-референса
  (русские метки), а не из C++-демо.

<details>
<summary><b>Python-референс (оригинальный README)</b></summary>

Тестовые изображения: `1.png`, `2.jpg`, `3.jpg`; метки `{"кот", "медведь", "лиса"}`.

```
0.8879  0.0063  0.1058
0.0014  0.0026  0.9960
0.0002  0.9994  0.0003
```
</details>

## Приложение: экспорт `jit_model.zip` для Torch-бэкенда

Для моделей без готового TorchScript-архива его можно получить из
[ai-forever/ru-clip](https://github.com/ai-forever/ru-clip):

```python
import ruclip, torch
from PIL import Image

def save_clip_model(clip_name: str, size: int, device: str):
    clip, processor = ruclip.load(clip_name, device=device)
    var = (torch.ones((1, 77)).long(), torch.ones((1, 3, size, size)))
    traced = torch.jit.trace(clip, var)
    traced.save("saves/" + clip_name + ".zip")

save_clip_model('ruclip-vit-large-patch14-336', 336, 'cpu')
```

## Лицензия

См. [LICENSE.txt](../CLIP/docs/ru-clip-main/LICENSE.txt) оригинального репозитория ru-clip.

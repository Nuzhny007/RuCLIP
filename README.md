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
| 3 | `TensorRTBackend` (engine из ONNX, кэш рядом с ONNX: `clip_*.{fp32\|fp16}.gpu<N>.b<B>.engine`) |
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
cmake --build rel --config Release
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

## Использование как библиотеки (CMake)

Проект собирается в **статическую библиотеку** `ruclip` (CMake-цель `ruclip::ruclip`) и
опциональную CLI-утилиту `RuCLIP` (`-DRUCLIP_BUILD_CLI=OFF` — только библиотека).
Библиотека экспортирует CMake-пакет `ruCLIP`: стороннему приложению достаточно
`find_package(ruCLIP)` и одной цели.

```bat
cmake -S RuCLIP -B build_lib -G "Visual Studio 17 2022" -A x64 ^
  -DRUCLIP_WITH_TORCH=ON -DRUCLIP_BUILD_CLI=OFF ^
  -DCMAKE_INSTALL_PREFIX=C:/work/clip/ruclip-install ^
  "-DTorch_DIR=C:/work/libraries/torch/libtorch-win-shared-with-deps-latest+cu126/libtorch/share/cmake/Torch" ^
  "-DOpenCV_DIR=C:/work/libraries/opencv/5.0/build_cuda/install/x64/vc17/lib"
cmake --build build_lib --config Release
cmake --install build_lib --config Release
```

Установка раскладывает:

```
<prefix>/lib/ruclip.lib
<prefix>/include/ruclip/...                # публичные заголовки
<prefix>/lib/cmake/ruCLIP/ruCLIPConfig.cmake
<prefix>/lib/cmake/ruCLIP/ruCLIPConfigVersion.cmake
<prefix>/lib/cmake/ruCLIP/ruCLIPTargets.cmake
```

Подключение в стороннем проекте:

```cmake
find_package(ruCLIP CONFIG REQUIRED)          # или -DruCLIP_DIR=<prefix>/lib/cmake/ruCLIP
target_link_libraries(app PRIVATE ruclip::ruclip)
```

Публичный API — `#include <ruclip/ruclip.h>`; высокоуровневый фасад `ruclip::Model`:

```cpp
#include <ruclip/ruclip.h>

ruclip::ModelOptions opt;
opt.modelDir = "C:/models/ruclip-vit-large-patch14-336";
opt.backend  = ruclip::BackendKind::TensorRT;   // Torch | TensorRT | Dnn
opt.useFp16  = true;

ruclip::Model model(opt);
ruclip::TensorF32 probs = model.ProbsPerImage({image}, {"кот", "медведь", "лиса"});
```

Конфиг-пакет сам вызывает `find_dependency` для OpenCV (обязательно), а также Torch и
CUDAToolkit — если библиотека собрана с этими бэкендами. Готовый пример-потребитель:
[`examples/basic`](examples/basic/README.md).

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

## TensorRT: генерация engine и кэш

Python и `trtexec` для инференса **не нужны**: engine собирает сама библиотека
(`nvinfer1::IBuilder` + ONNX-parser) при первом запуске.

Перед сборкой проверяется наличие файла рядом с ONNX:

```
<model>/clip_visual.<precision>.gpu<N>.b<B>.engine
<model>/clip_textual.<precision>.gpu<N>.b<B>.engine
```

- `<precision>` — `fp32` или `fp16` (флаг `--fp16`);
- `gpu<N>` — фактический индекс CUDA-устройства (`BackendConfig::deviceIndex`);
- `b<B>` — `maxBatch` optimization profile (флаг `--max_batch`).

Примеры: `clip_textual.fp16.gpu0.b8.engine`, `clip_visual.fp32.gpu1.b1.engine`.

Поведение:

1. файла нет → engine собирается из ONNX и сохраняется рядом с ONNX;
2. файл есть → engine читается с диска (сборки нет);
3. файл есть, но не десериализуется (другая версия TensorRT/GPU) → пересобирается
   и перезаписывается.

Все параметры, влияющие на содержимое engine, входят в имя, поэтому разные
конфигурации не перетирают кэш друг друга. Каталог `<model>/engines/`, который
использовали прежние версии, больше не читается и не создаётся.

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
| `--preprocess` | `letterbox` (по умолчанию), `centercrop` или `stretch` |
| `--text_file` | UTF-8-файл с метками (по одной на строку); перекрывает `--text` |
| `--max_batch` | верхняя граница batch для optimization profile TensorRT |
| `--fp16` | FP16: TensorRT (FP16-engine) и LibTorch (только CUDA); DNN игнорирует |
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

Сравнить бэкенды (включая FP16 для TensorRT и LibTorch) одним запуском:

```bat
python tools\run_bench.py --exe ..\rel_all\Release\RuCLIP.exe ^
    --clip ..\CLIP\data\ruclip-vit-large-patch14-336 --repeat 10 ^
    --backends torch,trt,trt:fp16,torch:fp16,dnn
```

Формат `backend:fp16` означает `--backend=<backend> --fp16=1`; для `Relevancy` добавьте
`--test-ind 1` (фазы: `reference`, `others`, `text`, `relevancy`).

Пример (large-336, 3 изображения × 3 метки, RTX 4070; `rel_all`):

```
backend       torch  total, ms       vs torch
---------------------------------------------
torch                   181.03          1.00x
trt                     104.62          1.73x
trt_fp16                 34.87          5.19x
torch_fp16               80.51          2.25x
dnn                    1908.98          0.09x

По фазам, avg ms:
backend     preprocess   encode_text  encode_image   postprocess         total
torch             4.28          7.06        169.66          0.03        181.03
trt               4.34          4.58         95.67          0.03        104.62
trt_fp16          4.25          4.78         25.82          0.02         34.87
torch_fp16        4.26          6.75         69.47          0.03         80.51
dnn               4.54         91.17       1813.24          0.04       1908.98
```

- **FP16 поддерживают TensorRT и LibTorch** (только CUDA; на CPU LibTorch FP16 не работает).
  OpenCV DNN флаг игнорирует.
- **TensorRT FP16** — ~5.2× к torch и ~3× к FP32-движку TensorRT; **LibTorch FP16** — ~2.25×
  к FP32 (основной выигрыш в `encode_image`: 69 против 170 мс).
- **Точность FP16** (`max|diff|` вероятностей относительно torch): LibTorch ≈ `2.7e-4`,
  TensorRT ≈ `2.6e-3` (у FP32-TRT `4.8e-4`, у DNN `5e-6`). Паритет-тест использует для FP16
  отдельный допуск `--tol-fp16 5e-3`.
- Для FP16 в LibTorch `RCLayerNorm` выполняет `LayerNorm` в float32 (веса приводятся к float32),
  иначе возникает ошибка смешанных типов.
- Время сборки engine и загрузки моделей/графов в замер не входит (конструктор бэкенда).
  DNN в OpenCV 5.1.0 сейчас CPU-only. Абсолютные числа зависят от модели, GPU и текущей
  загрузки — сравнивайте на своей машине.

## Структура проекта

```
RuCLIP/
├── CMakeLists.txt              # библиотека ruclip + CLI, install/export пакета ruCLIP
├── CMakeLists.Files.txt        # условные списки исходников
├── cmake/
│   └── ruCLIPConfig.cmake.in   # шаблон find_package(ruCLIP)
├── include/ruclip/             # ПУБЛИЧНЫЕ заголовки
│   ├── ruclip.h                # umbrella-заголовок
│   ├── Model.h                 # фасад ruclip::Model
│   ├── RuCLIPProcessor.h       # токенизация + препроцессинг
│   ├── Backend/
│   │   ├── ClipTypes.h         # TensorF32 / TextBatch / ImageBatch / BackendConfig
│   │   ├── ClipMath.h          # Logits / ProbsPerImage / Relevancy / GatherEos
│   │   ├── IClipBackend.h      # интерфейс + CreateBackend
│   │   ├── TorchBackend.h
│   │   ├── TensorRTBackend.h
│   │   └── DnnBackend.h
│   ├── ClipModel.h             # LibTorch-модель (CLIPImpl, внутренний)
│   ├── TextLower.h
│   ├── json*.hpp               # vendored nlohmann/json
│   └── youtokentome/           # vendored BPE
├── src/
│   ├── Model.cpp               # реализация фасада
│   ├── RuCLIP.cpp              # LibTorch-модель
│   ├── RuCLIPProcessor.cpp
│   ├── main.cpp                # CLI
│   ├── Backend/                # *.cpp бэкендов + BackendFactory
│   └── youtokentome/           # *.cpp BPE
├── examples/basic/             # приложение-потребитель find_package(ruCLIP)
├── tools/
│   ├── export_onnx.py          # экспорт clip_visual/clip_textual + logit_scale
│   ├── run_parity.py           # паритет-тест бэкендов
│   └── run_bench.py            # замеры производительности бэкендов
├── tests/                      # unit-тест ClipMath
└── docs/                       # архитектура, roadmap, дизайн бэкендов
```

## Известные ограничения и расхождения

- **TensorRT** — только NVIDIA GPU; engine привязан к GPU/версии TensorRT (пересобирается
  автоматически рядом с ONNX; имя включает точность, GPU и maxBatch); по умолчанию TF32
  (для строгого FP32 — `clearFlag(kTF32)`).
- **OpenCV DNN** — требует OpenCV 5.x; новый движок `ENGINE_OPENCV` сейчас CPU-only;
  `opencv_dnn` собран с CUDA, поэтому в PATH нужны cuDNN/ONNX Runtime.
- **LibTorch** — требует `jit_model.zip`; для base-моделей его надо экспортировать. FP16
  доступен только на CUDA (на CPU LibTorch FP16 не поддерживается).
- **Препроцессинг (исправлено)**: `cv::imread` возвращает **BGR**, а константы CLIP
  (mean/std) заданы в порядке **RGB**. Теперь изображение конвертируется `BGR→RGB` перед
  нормализацией — как в коммите `62a889c` оригинального репозитория. Режим
  `--preprocess=stretch` повторяет исходный `main.cpp`, а `--text_file` читает метки из
  UTF-8-файла (обход кодировки argv на Windows). После фикса C++ даёт
  `0.8798 / 0.0044 / 0.1158 …` (stretch + русские метки) — близко к Python-референсу ниже;
  остаточное расхождение ~0.01 — из-за разной интерполяции resize и декодера JPEG.

<details>
<summary><b>Python-референс (оригинальный README)</b></summary>

Тестовые изображения: `1.png`, `2.jpg`, `3.jpg`; метки `{"кот", "медведь", "лиса"}`.

```
0.8879  0.0063  0.1058
0.0014  0.0026  0.9960
0.0002  0.9994  0.0003
```

Проверка после RGB-фикса (stretch, метки из `tools/labels_ru.txt`):
```
0.8798  0.0044  0.1158
0.0000  0.0001  0.9999
0.0003  0.9994  0.0003
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

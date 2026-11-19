# Дизайн-документ: многобэкендная архитектура RuCLIP (C++)

> Проект: неофициальный C++-порт ruCLIP (Sber AI).
> Корень: `C:\work\clip\ruclip\RuCLIP`. Сборка: `C:\work\clip\ruclip\rel`.
> Статус: **реализовано** (Фазы 0–5). Актуальный обзор и инструкции — в `README.md`.

---

## 1. Цели и рамки

**Цель.** Разделить инференс ruCLIP на независимые бэкенды, чтобы:

1. собирать проект сразу с несколькими бэкендами и выбирать нужный в рантайме;
2. собирать проект с любым **одним** бэкендом (LibTorch, TensorRT или OpenCV DNN);
3. сохранить поведение и численные результаты текущей LibTorch-реализации как эталон.

**Бэкенды:**

| Бэкенд | Ключ | Артефакт модели | Аппаратное требование |
|---|---|---|---|
| LibTorch (текущий) | `torch` | `jit_model.zip` (TorchScript) | CPU / CUDA |
| TensorRT | `trt` | `clip_visual.onnx` + `clip_textual.onnx` → кэш `clip_*.{fp32\|fp16}.gpu<N>.b<B>.engine` | NVIDIA GPU + CUDA + TensorRT |
| OpenCV DNN | `dnn` | `clip_visual.onnx` + `clip_textual.onnx` | CPU (опц. OpenVINO/CUDA-плагин) |

**Не цели (вне рамок):**
- обучение / дообучение;
- добавление новых моделей или изменение архитектуры;
- async/стриминг-интерфейс для GPU (можно добавить позже, интерфейс это позволит);
- порт `RuCLIPPredictor` (roadmap 4.5).

---

## 2. Текущее состояние: что переиспользуемо, а что меняется

Сейчас функционал (`src/`) делится на четыре части; **переносить нужно только одну**.

| Часть | Файлы | Зависимость от LibTorch | В целевой архитектуре |
|---|---|---|---|
| Токенизация BPE + `lower()` | `RuCLIPProcessor.cpp`, `youtokentome/`, `TextLower.h` | нет | **без изменений** |
| Препроцессинг изображений | `RuCLIPProcessor.cpp` | OpenCV (+ `torch::Tensor` как носитель) | **без изменений**, выход в нейтральном буфере |
| Прямой проход модели (ViT + text-transformer) | `RuCLIP.cpp` | **да** | выносится за интерфейс `IClipBackend` |
| Постобработка (L2-norm, logits, softmax, `Relevancy`) | `RuCLIPProcessor.h`, `main.cpp` | написана на `torch::Tensor`, но это простые matmul/softmax | **переписывается** на нейтральные структуры |

Ключевой вывод: токенизатор и препроцессинг — это CPU-код, не зависящий от инференс-рантайма. Переносится только `CLIPImpl::EncodeImage/EncodeText/forward` (`src/RuCLIP.cpp:192-231`), а постобработка — это ~40 строк элементарной линейной алгебры, которую выгодно сделать общей для всех бэкендов.

Важное наблюдение: `EncodeImage` и `EncodeText` уже выделены как отдельные методы (`src/RuCLIP.cpp:192, 199`). Это ровно та декомпозиция, которая нужна для экспорта двух независимых графов: image-encoder и text-encoder.

---

## 3. Ключевое архитектурное решение: единый контракт данных

Все бэкенды общаются с остальным кодом через **нейтральный CPU-контракт**, не через `torch::Tensor`.

### 3.1 Типы данных (нейтральные)

```cpp
// Минимальный CPU-тензор float32 (row-major), общий для всех бэкендов.
// Форма произвольная: 2D достаточно для эмбеддингов и логитов,
// а [B, P, 2] требуется для Relevancy.
struct TensorF32 {
    std::vector<float> data;      // row-major
    std::vector<int64_t> shape;   // произвольная размерность
    int64_t rows() const;         // shape[0]
    int64_t cols() const;         // shape[1]
};

// Токены: [B * L] int32 + позиция EOS для каждой строки.
// int32, а не int64: токены <= 49408, и это снимает проблему int64-Gather
// у OpenCV DNN (см. раздел 7).
struct TextBatch {
    std::vector<int32_t> ids;      // [B * L], порядок bos .. eos .. pad(0)
    std::vector<int32_t> eos_pos;  // [B], индекс EOS внутри L=77
    int B = 0;
    int L = 0;
};

// Изображения: [B * 3 * H * W] float32, уже (HWC->CHW, /255, normalize mean/std).
struct ImageBatch {
    std::vector<float> pixels;     // [B * C * H * W]
    int B = 0, C = 0, H = 0, W = 0;
};
```

### 3.2 Контракт входов/выходов модели

| Метод | Вход | Выход | Примечание |
|---|---|---|---|
| `EncodeImage` | `ImageBatch` `[B,3,H,W]` f32, нормализован | `[B, D]` f32, **без** L2-нормы | D = `embed_dim` (512/768) |
| `EncodeText` | `TextBatch` `[B,77]` i32 | `[B, D]` f32, **без** L2-нормы | эмбеддинг на позиции EOS |

H = W = `image_resolution` (336/384); L = `context_length` (77); D = `embed_dim`.

### 3.3 Что НЕ входит в контракт модели

- **L2-нормализация** эмбеддингов — делается в общей постобработке;
- **`logit_scale`** и `logits = scale * (I_norm @ T_normᵀ)` — общая постобработка;
- **softmax по меткам** и **`Relevancy`** — общая постобработка.

Это гарантирует, что все бэкенды получают **идентичный** вход (препроцессинг один и тот же) и **идентичную** постобработку, т.е. расхождения могут возникать только в численной точности самого инференса.

---

## 4. Целевая архитектура

```
┌───────────────────────────────────────────────────────────────────┐
│  main.cpp (CLI)                                                     │
│    --backend=torch|trt|dnn   --clip=<model_dir>  ...               │
└───────────────┬───────────────────────────────────────────────────┘
                │ CreateBackend(kind, cfg)
                ▼
┌───────────────────────────────────────────────────────────────────┐
│  IClipBackend (интерфейс, backend-neutral)                         │
│    TensorF32 EncodeImage(const ImageBatch&);                       │
│    TensorF32 EncodeText(const TextBatch&);                         │
└──────┬──────────────────┬───────────────────────┬──────────────────┘
       │                  │                       │
  TorchBackend      TensorRTBackend            DnnBackend
  (jit_model.zip)   (.engine из ONNX)          (.onnx, cv::dnn)
       │                  │                       │
       └──────────┬───────┴───────────────────────┘
                  ▼
   Общий код (backend-neutral, без torch::Tensor):
   RuCLIPTokenizer / ImagePreprocessor / PostProcess / Relevancy / GatherEos
```

### 4.1 Интерфейс

```cpp
enum class BackendKind { Torch, TensorRT, Dnn };

struct BackendConfig {
    std::filesystem::path modelDir;   // каталог модели (config.json, bpe.model, ...)
    int deviceIndex = 0;              // для CUDA/TRT
    bool useFp16 = false;             // TRT: построить FP16-engine
    // ... (логгер, кэш engine и т.п.)
};

class IClipBackend {
public:
    virtual ~IClipBackend() = default;
    virtual BackendKind kind() const = 0;
    virtual TensorF32 EncodeImage(const ImageBatch& img) = 0;  // [B,D]
    virtual TensorF32 EncodeText(const TextBatch& txt) = 0;    // [B,D]
};

std::unique_ptr<IClipBackend> CreateBackend(BackendKind kind, const BackendConfig& cfg);
```

Почему CPU-контракт (вход и выход `std::vector<float>` / `TensorF32`), а не GPU-буферы:
- единый, тривиально проверяемый контракт для всех трёх бэкендов (включая CPU-OpenCV DNN);
- простое сравнение результатов между бэкендами (приёмочный тест);
- для текущего CLI-сценария (картинки + метки, батчи малы) оверхед копирования пренебрежим;
- асинхронный GPU-интерфейс можно добавить позже как отдельный метод, не ломая этот.

---

## 5. Общий код (переиспользуемый между бэкендами)

### 5.1 Процессор — источник нейтральных данных

`RuCLIPProcessor` остаётся, но его публичные методы, возвращающие `torch::Tensor`, дополняются/заменяются нейтральными, заполняющими `TextBatch`/`ImageBatch`:

- `EncodeText` → токенизация BPE + `lower()` + `bos/eos/pad` (логика `RuCLIPProcessor.cpp:48-69`), затем:
  - `ids` заполняется int32-токенами длины 77;
  - `eos_pos` = позиция вставленного `eos` (= число реальных токенов + 1, см. `RuCLIPProcessor.cpp:63-66`).
- `EncodeImage` → `ResizeToInput` (letterbox/center-crop, `RuCLIPProcessor.cpp:72-118`) + HWC→CHW + `/255` + `Normalize(mean,std)` — ровно текущая логика `RuCLIPProcessor.cpp:121-126`, но на выходе `ImageBatch`.

Это сохраняет препроцессинг **идентичным** для всех бэкендов и избавляет его от `torch::Tensor`.

> Текущие `torch::Tensor`-версии методов можно оставить как тонкие обёртки над нейтральными, чтобы не ломать существующий Torch-путь в первый момент.

### 5.2 Постобработка и `Relevancy`

Порт на `TensorF32` (без `torch::Tensor`). Требуется минимум примитивов:

```cpp
TensorF32 L2NormalizeRows(const TensorF32&);              // x /= ||x||_2 (по строкам)
TensorF32 MatMul(const TensorF32& A, const TensorF32& B); // A @ B
TensorF32 Transpose(const TensorF32& A);
TensorF32 SoftmaxRows(const TensorF32& A, float temp = 1.f);
TensorF32 GatherRows(const TensorF32& A, const std::vector<int32_t>& rows); // выбор строк
```

На их основе собираются:

- `Logits(I, T, logit_scale)` = `logit_scale * (L2NormalizeRows(I) @ L2NormalizeRows(T)ᵀ)`;
- `ProbsPerImage` = `SoftmaxRows(logits_per_image)`;
- `Relevancy(embeds, positives, negatives)` — прямой перенос текущей `inline Relevancy` из `RuCLIPProcessor.h:126-158` (softmax над парами, argmin «самого сложного» негатива). Формула не меняется, меняется только носитель.

> **Важно (`logit_scale`).** В текущем коде `LogitScale` — это параметр, загружаемый из весов (`src/RuCLIP.cpp:160,227`), инициализированный `ln(1/0.07)`, т.е. `exp(logit_scale) ≈ 14.2857`. Поскольку для новых бэкендов мы экспортируем только `EncodeImage`/`EncodeText` (без `forward`), `logit_scale` надо вынести наружу: записать его значение в `config.json` при экспорте (или sidecar-файл), а в постобработке читать оттуда. Жёстко зашивать `1/0.07` допустимо только как фолбэк с предупреждением.

### 5.3 `GatherEos` — общий помощник для TRT/DNN

Чтобы не делать динамический `Gather` по позиции EOS внутри графа (рискованно для OpenCV DNN, см. раздел 7), text-encoder экспортируется как «полная последовательность»:

```
clip_textual.onnx:  input  input_ids [B, 77] i32
                    output text_features [B, 77, D] f32   (ln_final + text_projection по всем 77 позициям)
```

Выбор строки EOS выполняется снаружи, общим помощником:

```cpp
// full: [B*77*D]; eos_pos: [B]; -> [B*D]
std::vector<float> GatherEos(const std::vector<float>& full, int B, int L, int D,
                             const std::vector<int32_t>& eos_pos);
```

Математически это идентично текущему «взять эмбеддинг на позиции EOS, затем `TextProjection`», потому что `LayerNorm` и `TextProjection` позиционно-независимы (применяются к каждому из 77 токенов отдельно). Для Torch-бэкенда менять ничего не нужно — он продолжает использовать текущую реализацию с `torch::where` (`src/RuCLIP.cpp:208`).

---

## 6. Артефакты модели и экспорт ONNX

### 6.1 Раскладка каталога модели

```
ruclip-vit-<variant>/
├── config.json              # размерности + mean/std (+ logit_scale после экспорта)
├── bpe.model                # токенизатор (общий)
├── jit_model.zip            # Torch-бэкенд (существующий)
├── clip_visual.onnx         # image-encoder (для trt и dnn)
├── clip_textual.onnx        # text-encoder, полная последовательность [B,77,D]
├── clip_visual.<fp32|fp16>.gpu<N>.b<B>.engine   # кэш TensorRT, рядом с ONNX
└── clip_textual.<fp32|fp16>.gpu<N>.b<B>.engine
```

Имя engine включает все параметры, влияющие на содержимое: точность (`fp32`/`fp16`),
фактический индекс CUDA-устройства (`gpu0`, `gpu1`, …) и `maxBatch` (`b1`, `b8`, …).
Пример: `clip_textual.fp16.gpu0.b8.engine`.

### 6.2 Экспорт (одноразовый, Python, офлайн)

Скрипт: [`tools/export_onnx.py`](../tools/export_onnx.py). Экспорт делается **из оригинальной Python-модели `ruclip`**, а не из C++-порта (C++ не нужен на этом шаге). Ноутбук `docs\ru-clip-main\jupyters\RuCLIP_onnx_example.ipynb` (`clip_onnx`, Lednik7/CLIP-ONNX) — альтернативный путь.

Порядок экспорта (реализовано):

1. Загрузка `ruclip.model.CLIP.from_pretrained(<каталог модели>)` — **без сети**, только локальные `config.json`/`pytorch_model.bin`.
2. `EncodeImage`-подграф: `input_names=["pixel_values"]`, `output_names=["image_features"]`, вход `[1,3,H,W]` f32.
3. `EncodeText`-подграф: `input_names=["input_ids"]`, `output_names=["text_features"]`, вход `[1,77]` **int32**, выход **полная последовательность** `[1,77,D]` (`ln_final` → `text_projection` по всем позициям, без среза по EOS).
4. `opset_version = 14`: `torch>=2.x` экспортирует multi-head attention через `aten::scaled_dot_product_attention`, который поддержан только с opset 14; граф при этом **полностью разложен** на стандартные операции (`MatMul`/`Gemm`, `Softmax`, `ReduceMean`/`Sqrt`/`Pow`/`Div` для LayerNorm, `Sigmoid`+`Mul` для QuickGELU, `Gather`), без слитых `Attention`/`LayerNormalization`.
5. Динамический batch (ось 0) — для этого в обёртках используются `flatten(2)` и `zeros_like` (иначе batch запекается константой).
6. Дописать `logit_scale = exp(logit_scale)` в `config.json`.

> **`logit_scale` обучаемый и различается по моделям** — на трёх моделях получено `31.40`, `45.29`, `48.74` (а не стандартные `1/0.07 ≈ 14.29`). Поэтому его обязательно читать из `config.json`, а не хардкодить.

> `QuickGELU = x·σ(1.702x)` не является стандартным оператором ни в ONNX, ни в TensorRT/OpenCV. Экспортёр раскладывает его в `Sigmoid + Mul` (в графе присутствуют оба узла).

Проверка (скрипт делает автоматически, `onnxruntime`): max|diff| PyTorch vs ONNX ≈ `1e-7…1e-5` для batch=1 и batch=3 на всех трёх моделях.

---

## 7. Дизайн бэкендов

### 7.1 TensorRTBackend ✅ (низкий риск)

**Путь данных:** ONNX → engine (`trtexec` офлайн или `nvinfer1::IBuilder` + ONNX-parser в C++) → `IExecutionContext::enqueueV3`.

**Поддержка операторов модели:**

| Операция модели | В TensorRT | Комментарий |
|---|---|---|
| `Conv2d` (patch-embedding, stride=patch, bias=false) | ✅ | стандарт |
| `Linear`/`MatMul` (attention, MLP, проекции) | ✅ | стандарт |
| `LayerNorm` (fp16-safe) | ✅ | нативно; при необходимости разворачивается в примитивы |
| `QuickGELU` | ✅ | `Sigmoid + Mul` |
| Self-attention + статическая каузальная маска (`-inf + triu`) | ✅ | маска — константа в графе |
| `Embedding` (49408 × D) | ✅ | `Gather` по таблице весов |
| Динамический batch | ✅ | optimization profile (kMIN/kOPT/kMAX) |
| Динамический `Gather` по EOS | ⚠️ | поддерживается, но **исключён из графа** (см. 5.3) |

**Ключевые решения:**
- **Формы входа статичны, кроме batch.** Текст `[B,77]`, картинка `[B,3,H,W]`. Для engine задаётся optimization profile по `B` (например, `[1..64]`). Это самый простой сценарий для TRT.
- **`input_ids` в engine как int32** (надёжнее int64 в TRT), приведение из `TextBatch::ids` тривиально.
- **Сборка engine**: при первом запуске из ONNX (кэш рядом с ONNX: `clip_*.{fp32|fp16}.gpu<N>.b<B>.engine`); Python/`trtexec` не нужны. В конфиге — флаг `useFp16` и `maxBatch` (оба входят в имя).
- **Исполнение**: для простоты — синхронно: `cudaMemcpy(H2D)` → `enqueueV3` → `cudaMemcpy(D2H)`. Выход text-энкодера `[B,77,D]` пропускается через общий `GatherEos`.

**Ограничения (осознанно принимаются):**
- только NVIDIA GPU (CUDA + cuDNN + TensorRT той же версии);
- engine аппаратно-зависим (пересборка под GPU/версию TRT — офлайн-шаг);
- нет CPU-fallback: на машине без GPU бэкенд недоступен (это не ошибка сборки, а рантайм-ограничение).

**Реализация (Фаза 3):** `src/Backend/TensorRTBackend.cpp` (~400 строк).

- Окружение машины: **TensorRT 10.16.1.11**, **CUDA 13.2**, драйвер 621, RTX 4070 (sm89), OpenCV 4.14.
- Движки собираются из ONNX при первом запуске и кэшируются **рядом с ONNX**:
  `clip_visual.<fp32|fp16>.gpu<N>.b<B>.engine` и `clip_textual.<fp32|fp16>.gpu<N>.b<B>.engine`
  (для large-336 fp32 ≈ 1.2 ГБ + 0.5 ГБ). Python/`trtexec` в рантайме не нужны — engine
  строит сама библиотека через `IBuilder` + ONNX-parser.
- При последующих запусках engine читается с диска. Если кэш несовместим с текущей
  версией TensorRT/GPU, он пересобирается и перезаписывается автоматически.
- CMake: `RUCLIP_WITH_TRT=ON`, `RUCLIP_TRT_ROOT` (автоопределение `C:/work/libraries/cuda/TensorRT-*`), `find_package(CUDAToolkit)`.
- Особенности TRT 10.16: интерфейсы освобождаются через `delete` (`destroy()` удалён), у `IOptimizationProfile` protected-деструктор (владение забирает `addOptimizationProfile`).
- Совместимость с torch (cu126): torch ожидает `CUDA::nvToolsExt`, которого нет в CUDA 13 — в CMake добавляется пустая INTERFACE-цель.
- TF32 в TRT включён по умолчанию: расхождение с torch ≈ 5e-4 (для строгого FP32 — `config->clearFlag(BuilderFlag::kTF32)`).

**Проверено:**
- `--backend=trt` (large-336): `0.8171 / 0.00347 / 0.1794 …` против torch `0.8166 / 0.00347 / 0.1799 …`;
- `Relevancy` (test_ind=1) совпадает с torch до ~5e-5;
- мультибэкендная сборка (torch + TRT в одном exe) собирается, и оба `--backend=torch|trt` работают.

---

### 7.2 DnnBackend ✅ (реализовано на OpenCV 5.1.0)

**Путь данных:** `cv::dnn::readNetFromONNX(path, cv::dnn::ENGINE_OPENCV)` → `net.setInput` → `net.forward`.

**Реализация (Фаза 4):** `src/Backend/DnnBackend.cpp` (~100 строк).

- Требует **OpenCV 5.x** (в окружении — 5.1.0, `OPENCV_DNN_CUDA=ON`). Новый graph-движок `ENGINE_OPENCV` разбирает весь ONNX-граф (opset 14, разложенный): 24-слойный ViT и текстовый трансформер с `LayerNorm`, `QuickGELU`, каузальной маской и int32-`Embedding`.
- `input_ids` подаётся как `cv::Mat` `CV_32S` `[B,77]`; изображения — `cv::Mat` `CV_32F` `[B,3,H,W]`. `setInput` копирует данные, поэтому внешние буферы `ImageBatch`/`TextBatch` можно передавать как non-owning `cv::Mat`.
- Динамический batch работает (проверено на `B=3`).
- `readNetFromONNX` — с явным `ENGINE_OPENCV` (не `ENGINE_AUTO`, который может смениться).
- `logit_scale` — из `config.json`; EOS — общий `GatherEos`.

**Проверено (large-336, один exe с torch/TRT):**
- `--backend=dnn`: `0.81661 / 0.00347 / 0.17992 …` против torch `0.816615 / 0.00347 / 0.179912 …` (расхождение ~1e-5);
- `Relevancy` (test_ind=1): `0.997076 / 0.996927 / 0.997114 / 0.999034 / 0.988056` против torch `0.997069 / …` (~5e-5).

**Окружение/запуск:**
- `find_package(OpenCV REQUIRED)` + проверка `>= 5.0` при `RUCLIP_WITH_DNN=ON`;
- `opencv_dnn510.dll` собран с CUDA/ONNX Runtime, поэтому в `PATH` нужны `CUDA\v13.2\bin\x64` (cublas/cudart), `cudnn64_9.dll` и `onnxruntime.dll`;
- готовый скрипт — `rel_all\Release\run_all.bat torch|trt|dnn`.

> 💡 ONNX Runtime остаётся разумной альтернативой, если нужен CPU-рантайм без тяжёлой сборки OpenCV 5.x, но на практике OpenCV 5.1.0 справился с графом без обходов: разложение `LayerNorm`/`QuickGELU` уже сделано на этапе экспорта (opset 14), int32-токены и вынос EOS-`Gather` — тоже часть контракта.

---

## 8. Система сборки (CMake)

```
option(RUCLIP_WITH_TORCH "LibTorch backend"  ON)
option(RUCLIP_WITH_TRT   "TensorRT backend"  OFF)
option(RUCLIP_WITH_DNN   "OpenCV DNN backend" OFF)
```

Зависимости подключаются условно:

- **OpenCV (обязательно, для I/O):** `find_package(OpenCV 4.6 REQUIRED COMPONENTS core imgproc highgui)`.
- **Torch** — только при `RUCLIP_WITH_TORCH`: `find_package(Torch REQUIRED)`.
- **TensorRT** — только при `RUCLIP_WITH_TRT`: `find_path(NvInfer.h)` + `find_library(nvinfer)` + `find_library(nvonnxparser)` + CUDA.
- **DNN** — только при `RUCLIP_WITH_DNN`: `find_package(OpenCV REQUIRED COMPONENTS dnn)` (версия ≥ 4.10/5.0 — с проверкой).

Фабрика компилируется всегда; конкретные `*Backend.cpp` включаются по флагам. При `RUCLIP_WITH_*` = только один бэкенд лишние исходники просто не попадают в цель. Определения `RUCLIP_HAS_TORCH/TRT/DNN` (через `target_compile_definitions`) дают в коде `#ifdef`-ветки и корректные сообщения об ошибке, если пользователь запросил `--backend=trt`, а сборка без TensorRT.

Заодно закрываем пункты roadmap 3.3/4.8: убрать абсолютный путь MKL (`CMakeLists.Files.txt:9-11`), добавить `find_package(Threads)`.

---

## 9. Выбор рантайма (CLI)

В `main.cpp` добавляется ключ `--backend` (по аналогии с существующими `--test_ind`/`--preprocess` в `main.cpp:186-194`):

```
--backend=torch|trt|dnn   (по умолчанию torch)
```

Логика:
1. распарсить `--backend` в `BackendKind`;
2. проверить, что этот бэкенд скомпилирован (`RUCLIP_HAS_*`), иначе — понятная ошибка;
3. `CreateBackend(...)` → `processor` (общий) → два режима `MatchImageAndText` / `MatchOne2ManyImages`, где вместо `clip->forward/Encode*` вызываются методы интерфейса, а softmax/`Relevancy` берутся из общего кода.

---

## 10. План реализации по фазам

| Фаза | Содержание | Риск | Результат |
|---|---|---|---|
| **0** | Заморозить контракт (раздел 3); извлечь нейтральные `TextBatch`/`ImageBatch`/`TensorF32`; вынести постобработку и `Relevancy` на `TensorF32`; `GatherEos` | низкий | общий код, независимый от Torch |
| **1** | Ввести `IClipBackend` + `TorchBackend` (обёртка над текущим `CLIPImpl`); `--backend=torch`; CMake-опции | низкий | рефакторинг без изменения поведения |
| **2** | `tools/export_onnx.py`: экспорт `clip_visual.onnx` + `clip_textual.onnx` (+ `logit_scale` в config) для трёх моделей | низкий | артефакты ONNX |
| **3** | `TensorRTBackend`: построение engine + `enqueueV3` + `GatherEos`; `--backend=trt` | средний | GPU-бэкенд |
| **4** | `DnnBackend`: `readNetFromONNX` + развёрнутый LayerNorm + int32-токены; `--backend=dnn` | высокий | CPU-бэкенд |
| **5** | Смоук-тесты паритета (см. 11), документация, `.bat`-скрипты запуска | средний | готовая многобэкендная сборка |

Фазы 1 и 2 независимы и могут идти параллельно. Фаза 4 — после 2.

> **Статус:** Фазы 0–5 реализованы — нейтральный слой в `src/Backend/` (`ClipTypes.h`,
> `ClipMath.*`, `IClipBackend.h`, `BackendFactory.cpp`), `TorchBackend`, `TensorRTBackend`,
> `DnnBackend`, ключ `--backend`, CMake-опции `RUCLIP_WITH_TORCH/TRT/DNN`. Проект собирается
> и без LibTorch (`-DRUCLIP_WITH_TORCH=OFF`), и мультибэкендно: один exe с torch + TensorRT +
> OpenCV DNN, выбор через `--backend=torch|trt|dnn`. ONNX-артефакты (`clip_visual.onnx`,
> `clip_textual.onnx`) и `logit_scale` в `config.json` получены для трёх моделей скриптом
> `tools/export_onnx.py` (opset 14, int32-токены, dynamic batch) и проверены `onnxruntime`;
> TensorRT-engine собираются из ONNX и кэшируются рядом с ONNX (имя включает точность,
> GPU и maxBatch). Паритет бэкендов проверяется
> скриптом `tools/run_parity.py` (torch/trt/dnn, порог `2e-3`).

---

## 11. Приёмка: проверка паритета

**Автоматически:** [`tools/run_parity.py`](../tools/run_parity.py) прогоняет один exe с
`--backend=torch|trt|dnn` и сравнивает матрицу zero-shot и значения `Relevancy`
(эталон — torch, порог `2e-3`, коды возврата 0/1 — пригодно для CI).

Ручные/дополнительные проверки:
1. **Zero-shot**: у бэкендов совпадают вероятности с точностью до численных эффектов
   (TF32 в TRT ~5e-4, DNN ~1e-5). Числа `0.8879 / …` — Python-референс (русские метки);
   C++ их точно не воспроизводит из-за BGR/RGB и letterbox (см. `README.md`).
2. **`Relevancy`** (`test_ind=1`): формы `[B,·]`, `positives > 1` — паритет проверен.
3. **Сборка**: все комбинации флагов: только Torch, только TRT, только DNN, torch+TRT, все три.
4. **Юнит-тест `ClipMath`**: `tests/test_clip_math.cpp` (matmul/norm/softmax/`Relevancy`/`GatherEos`).

---

## 12. Риски и открытые вопросы

| # | Риск / вопрос | Влияние | Митигация |
|---|---|---|---|
| 1 | OpenCV DNN не покрывает какой-то узел графа (attention/LayerNorm) | блокер фазы 4 | разворачивать LayerNorm; поднять версию OpenCV до 5.0; фолбэк — ONNX Runtime |
| 2 | `logit_scale` — обучаемый параметр или константа? | точность лоgitов | вынести значение в `config.json` при экспорте; не хардкодить |
| 3 | fp16 в TRT даёт расхождение эмбеддингов | точность | fp32 по умолчанию; fp16 только за флагом + тест паритета |
| 4 | Engine TRT привязан к GPU/версии | переносимость | пересборка при первом запуске/несовместимом кэше; кэш рядом с ONNX, имя включает precision/gpu/batch |
| 5 | Динамический batch в OpenCV DNN ограничен | удобство | фиксировать batch или пересоздавать сеть |
| 6 | Нужен ли `DecodeText`/`Predictor` (roadmap 4.4/4.5) для новых бэкендов | объём | вне рамок этого дизайна; при необходимости — общий код, не бэкенд-специфичный |
| 7 | int64 vs int32 токенов и обратная совместимость Torch-пути | контракт | Torch-бэкенд приводит int32→int64 на входе в `torch::tensor(kLong)` |

---

## 13. Итог

Реализовано (Фазы 0–5):

- **TensorRT** — `TensorRTBackend`, engine из ONNX (кэш рядом с ONNX, имя `clip_*.{fp32|fp16}.gpu<N>.b<B>.engine`), `enqueueV3`, паритет с torch ~5e-4 (TF32).
- **OpenCV DNN** — `DnnBackend` на OpenCV 5.1.0 (новый движок `ENGINE_OPENCV`), паритет с torch ~1e-5; обошлось без ручного разворачивания операторов — сказалась подготовка экспорта (opset 14, int32, EOS-`Gather` вне графа).
- Единый интерфейс `IClipBackend` + общий препроцессинг/постобработка дают и «сборку со всеми бэкендами», и «сборку с одним»; поведение Torch-ветки сохранено (паритет со старым бинарником подтверждён).
- Добавлены `tools/export_onnx.py`, `tools/run_parity.py`, юнит-тест `ClipMath`, скрипты запуска и README.

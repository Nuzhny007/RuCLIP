# Дизайн-документ: многобэкендная архитектура RuCLIP (C++)

> Проект: неофициальный C++-порт ruCLIP (Sber AI).
> Корень: `C:\work\clip\ruclip\RuCLIP`. Сборка: `C:\work\clip\ruclip\rel`.
> Статус: **проект дизайна** — код ещё не изменён.

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
| TensorRT | `trt` | `clip_visual.engine` + `clip_textual.engine` (из ONNX) | NVIDIA GPU + CUDA + TensorRT |
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
// Минимальный CPU-тензор float32. Достаточен для эмбеддингов и логитов.
struct TensorF32 {
    std::vector<float> data;   // row-major
    int rows = 0;
    int cols = 0;
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
└── engines/                 # (только TensorRT, необязательно, кэш)
    ├── clip_visual.engine
    └── clip_textual.engine
```

### 6.2 Экспорт (одноразовый, Python, офлайн)

Экспорт делается **из оригинальной Python-модели `ruclip`**, а не из C++-порта (C++ не нужен на этом шаге). В репозитории уже есть пример пути — ноутбук `docs\ru-clip-main\jupyters\RuCLIP_onnx_example.ipynb` использует `clip_onnx` (Lednik7/CLIP-ONNX), который уже делит модель на `clip_visual.onnx` и `clip_textual.onnx`.

Порядок экспорта (скрипт будет добавлен в `tools/export_onnx.py`):

1. `ruclip.load(name)` → `clip`, `processor`.
2. `EncodeImage`-подграф: `torch.onnx.export` с `input_names=["pixel_values"]`, `output_names=["image_features"]`, вход `[1,3,H,W]` f32.
3. `EncodeText`-подграф: `input_names=["input_ids"]`, `output_names=["text_features"]`, вход `[1,77]` **int64→int32 на входе**, выход **полная последовательность** `[1,77,D]` (для этого берём `ln_final`-выход и применяем `text_projection` ко всем позициям, без среза по EOS).
4. `opset_version`: 13 (компромисс для OpenCV DNN) или 17 (если целимся только на TRT/ORT). Для DNN дополнительно — развёрнутый `LayerNorm` (см. 7.2).
5. Дописать `logit_scale` (и при необходимости `mean`/`std`) в `config.json`.

> `QuickGELU = x·σ(1.702x)` не является стандартным оператором ни в ONNX, ни в TensorRT/OpenCV. Экспортёр сам разложит его в `Sigmoid + Mul` — дополнительных действий не требуется, но в приёмочном тесте стоит явно проверить этот узел графа.

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
- **Сборка engine**: по умолчанию — при первом запуске из ONNX (кэшируется в `engines/`), альтернатива — офлайн `trtexec`. В конфиге — флаг `useFp16`.
- **Исполнение**: для простоты — синхронно: `cudaMemcpy(H2D)` → `enqueueV3` → `cudaMemcpy(D2H)`. Выход text-энкодера `[B,77,D]` пропускается через общий `GatherEos`.

**Ограничения (осознанно принимаются):**
- только NVIDIA GPU (CUDA + cuDNN + TensorRT той же версии);
- engine аппаратно-зависим (пересборка под GPU/версию TRT — офлайн-шаг);
- нет CPU-fallback: на машине без GPU бэкенд недоступен (это не ошибка сборки, а рантайм-ограничение).

**Оценка объёма кода:** `TensorRTBackend` + построение engine ≈ 300–500 строк, плюс RAII-обёртки над `nvinfer1`/`cuda` (деструкторы, `unique_ptr`-кастом-делетеры).

---

### 7.2 DnnBackend ⚠️ (реализуемо, но самый хрупкий)

**Путь данных:** `cv::dnn::readNetFromONNX` → `net.setInput` → `net.forward`.

**Поддержка операторов модели (ключевой риск):**

| Операция модели | В OpenCV DNN | Комментарий / обход |
|---|---|---|
| `Conv2d`, `Linear`/`MatMul`, `Softmax` | ✅ | стандарт |
| `QuickGELU` | ✅ | `Sigmoid + Mul` |
| Self-attention (reshape/transpose) | ⚠️ | паттерн многоголового внимания из PyTorch-экспорта — известное место падений shape-inference |
| `LayerNorm` | ⚠️ | нативно отсутствовал до поздних 4.x (≈4.9); полная поддержка трансформеров — в OpenCV 5.0 |
| `Embedding` (int64 `Gather`) | ⚠️ | int32/int64 поддержаны поздно; обход — int32-токены |
| Динамический `Gather` по EOS | ❌/⚠️ | **исключён из графа** (см. 5.3) |

**Ключевые решения и обходы:**

1. **Версия OpenCV.** Текущее требование `find_package(OpenCV 4.6)` (`CMakeLists.txt:10`) **недостаточно**: 4.6 не имеет нативного ONNX `LayerNorm` и стабильного int64-пути. Для DNN-бэкенда нужно ≥ 4.10, а для уверенной работы трансформера — фактически **OpenCV 5.0** (переписанный graph-движок, покрытие ~80% ONNX-операторов). Базовый OpenCV для I/O остаётся 4.6; модуль `dnn` тянет более новая версия.
2. **`LayerNorm`.** Два варианта:
   - экспорт ONNX на opset 11–13 с **развёрнутым** `LayerNorm` (`ReduceMean → Sub → Pow → ReduceMean → Sqrt → Div → Mul → Add`);
   - либо OpenCV с нативным `LayerNorm`.
   Рекомендация: разворачивать при экспорте — это убирает зависимость от версии OpenCV.
3. **`Embedding`/int64.** Токены идут int32 (контракт `TextBatch`), поэтому int64-`Gather` не возникает вовсе.
4. **Динамический EOS-`Gather`.** Полностью вынесен в `GatherEos` (см. 5.3) — самый проблемный динамический оператор исчезает из графа.
5. **Batch.** `cv::dnn::Net` поддерживает динамику ограниченно; проще фиксировать batch через `net.setInput` (или пересоздавать сеть под нужный `B`).

**Оценка объёма кода:** `DnnBackend` сам по себе мал (~100–150 строк), но **основные затраты — не код, а отладка несовместимости операторов** (подбор способа экспорта/opset/версии OpenCV) и приёмочный тест на паритет.

> 💡 **Если цель DNN — портативный CPU-рантайм без LibTorch**, рекомендую рассмотреть **ONNX Runtime** как альтернативу внутри того же интерфейса `IClipBackend`: тот же ONNX-файл, но полная поддержка `LayerNorm`/int64/динамических форм без боли OpenCV DNN. OpenCV DNN оправдан, когда «уже есть OpenCV и добавлять другой рантайм нельзя», либо с OpenCV 5.0.

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

---

## 11. Приёмка: проверка паритета

1. **Смоук-тест zero-shot** (эталон из `README.md:27-31`): картинки `кот/медведь/лиса` → вероятности
   `0.8879 / 0.0063 / 0.1058`, `0.0014 / 0.0026 / 0.9960`, `0.0002 / 0.9994 / 0.0003`.
   Прогон на `torch`, `trt`, `dnn` — результаты должны совпадать в пределах допуска.
2. **Паритет эмбеддингов**: на фиксированных входных данных сравнить `EncodeImage`/`EncodeText`
   между бэкендами (косинусная близость > 0.9999 для fp32; отдельный допуск для fp16-TRT).
3. **Паритет `Relevancy`**: те же случаи форм `[1,·]` и `[B,·]`, `positives.size() ∈ {1, >1}`
   (roadmap 4.7).
4. **Сборка**: проверить все комбинации флагов: только Torch, только TRT, только DNN, все три.

---

## 12. Риски и открытые вопросы

| # | Риск / вопрос | Влияние | Митигация |
|---|---|---|---|
| 1 | OpenCV DNN не покрывает какой-то узел графа (attention/LayerNorm) | блокер фазы 4 | разворачивать LayerNorm; поднять версию OpenCV до 5.0; фолбэк — ONNX Runtime |
| 2 | `logit_scale` — обучаемый параметр или константа? | точность лоgitов | вынести значение в `config.json` при экспорте; не хардкодить |
| 3 | fp16 в TRT даёт расхождение эмбеддингов | точность | fp32 по умолчанию; fp16 только за флагом + тест паритета |
| 4 | Engine TRT привязан к GPU/версии | переносимость | пересборка офлайн/при первом запуске; кэш в `engines/` |
| 5 | Динамический batch в OpenCV DNN ограничен | удобство | фиксировать batch или пересоздавать сеть |
| 6 | Нужен ли `DecodeText`/`Predictor` (roadmap 4.4/4.5) для новых бэкендов | объём | вне рамок этого дизайна; при необходимости — общий код, не бэкенд-специфичный |
| 7 | int64 vs int32 токенов и обратная совместимость Torch-пути | контракт | Torch-бэкенд приводит int32→int64 на входе в `torch::tensor(kLong)` |

---

## 13. Итог

- **TensorRT** — реализуем полностью, путь стандартный (ONNX → engine), риск низкий, максимальный прирост скорости. Рекомендуется первым.
- **OpenCV DNN** — реализуемо, но требует апгрейда OpenCV (≥4.10, фактически 5.0), разворачивания `LayerNorm`, int32-токенов и выноса динамического EOS-`Gather` из графа. Самый трудоёмкий и рискованный путь; разумная альтернатива для CPU-рантайма — ONNX Runtime.
- Единый интерфейс `IClipBackend` + общий препроцессинг/постобработка дают и «сборку со всеми бэкендами», и «сборку с одним», без изменения поведения текущей Torch-ветки.

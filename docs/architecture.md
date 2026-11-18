# Архитектура проекта RuCLIP (C++ / LibTorch)

> Неофициальный порт мультимодальной модели **ruCLIP** (Sber AI) с Python/PyTorch на **C++ / LibTorch**.
> Корень проекта: `C:\work\clip\ruclip\RuCLIP`. Сборка: `C:\work\clip\ruclip\rel`.

---

## 1. Назначение

Проект воспроизводит поведение `ru-clip` (https://github.com/ai-forever/ru-clip):

- получение совместных эмбеддингов изображений и текста;
- ранжирование «изображение ↔ текст» (zero-shot классификация);
- поиск соответствий между изображениями.

В отличие от оригинального `ru-clip`, в C++-порте дополнительно добавлена функция
`Relevancy(...)`, вычисляющая «релевантность» запроса относительно набора позитивных и
негативных эмбеддингов (см. раздел 8).

Поддерживаемые модели (см. `ruclip\CLIP\data\`):

| Модель | embed_dim | image_resolution | vision | text |
|---|---|---|---|---|
| `ruclip-vit-large-patch14-336` | 768 | 336 | L=24, W=1024, patch=14 | L=12, W=768, heads=12 |
| `ruclip-vit-base-patch16-384` | 512 | 384 | L=12, W=768, patch=16 | L=12, W=512, heads=8 |
| `ruclip-vit-base-patch32-384` | 512 | 384 | L=12, W=768, patch=32 | L=12, W=512, heads=8 |

---

## 2. Структура репозитория

```
RuCLIP/
├── CMakeLists.txt            # верхний CMake: ищет Torch + OpenCV, собирает exe
├── CMakeLists.Files.txt      # списки исходников/заголовков/библиотек
├── README.md                 # описание, примеры, инструкция экспорта весов
├── docs/                     # документация (этот файл + roadmap.md)
└── src/
    ├── TorchHeader.h         # обёртка include <torch/torch.h> с подавлением предупреждений MSVC
    ├── json.hpp / json_fwd.hpp   # vendored nlohmann::json
    ├── RuCLIP.h              # объявления классов модели + FromPretrained
    ├── RuCLIP.cpp            # реализация модели (Transformer, ViT, CLIP)
    ├── RuCLIPProcessor.h     # объявления процессора + inline Relevancy
    ├── RuCLIPProcessor.cpp   # токенизация, препроцессинг изображений
    ├── main.cpp              # CLI-демо: два режима сопоставления
    └── youtokentome/         # vendored токенизатор YouTokenToMe (BPE)
        ├── bpe.{h,cpp}
        ├── utf8.{h,cpp}
        ├── utils.{h,cpp}
        └── third_party/flat_hash_map.h
```

`rel/` — каталог сборки CMake (Visual Studio solution), `rel/Release/` — итоговый
`RuCLIP.exe` и `.bat`-сценарии запуска с прописанными путями к DLL.

---

## 3. Зависимости

- **LibTorch** (C++ API PyTorch) — `libtorch-win-shared-with-deps-latest+cu126`.
- **OpenCV 4.6+** — загрузка/ресайз изображений, отрисовка результатов.
- **YouTokenToMe** (vendored, пространство имён `vkcom`) — BPE-токенизатор, совместимый
  с Python-пакетом `youtokentome`.
- **nlohmann/json** (vendored) — чтение `config.json`.
- **Intel MKL** — в `CMakeLists.Files.txt` прописан абсолютный путь (см. roadmap).

---

## 4. Архитектура модели

Классы в `src/RuCLIP.h`/`src/RuCLIP.cpp` повторяют структуру `ru-clip/ruclip/model.py`.
Каждый C++-класс использует идиому `XxxImpl` + `TORCH_MODULE(Xxx)` (т.е. `Xxx` — это
`ModuleHolder<XxxImpl>`).

| Python (`ruclip/model.py`) | C++ (`src/RuCLIP.*`) | Назначение |
|---|---|---|
| `LayerNorm` (fp16-safe) | `RCLayerNorm` | LayerNorm с приведением к float32 и обратно |
| `QuickGELU` | `QuickGELU` | `x * sigmoid(1.702 * x)` |
| `ResidualAttentionBlock` | `ResidualAttentionBlock` | self-attention + MLP с остаточными связями |
| `Transformer` | `Transformer` | стек `ResidualAttentionBlock` |
| `VisionTransformer` | `VisionTransformer` | изображение → визуальный эмбеддинг |
| `CLIP` | `CLIP` | объединяет ViT и текстовый Transformer |

### 4.1 ResidualAttentionBlock (`src/RuCLIP.cpp:3-37`)

```
x → ln_1 → attention → (+x) → ln_2 → mlp → (+x)
```

- `attn = MultiheadAttention(d_model, n_head)` (batch_first=false, т.е. вход `[L,N,E]`).
- `mlp = Sequential(c_fc: Linear(d,4d) → gelu: QuickGELU → c_proj: Linear(4d,d))`.
- `AttnMask` хранится как обычное поле (не `register_buffer`, см. roadmap), при прямом
  проходе приводится к `x.dtype()`/`x.device()`.

### 4.2 Transformer (`src/RuCLIP.cpp:41-78`)

`Resblocks` — `torch::nn::Sequential` из `Layers` блоков. `forward` просто прогоняет
тензор через `Resblocks`. `InitializeParameters()` выполняет инициализацию «с нуля»
(не используется при загрузке готовых весов, но вызывается в конструкторе `CLIPImpl`).

### 4.3 VisionTransformer (`src/RuCLIP.cpp:82-129`)

```
x [N,3,H,W]
 → conv1 (patch_size×patch_size, stride=patch_size, bias=false)  [N,width,grid,grid]
 → reshape [N,width,grid²] → permute [N,grid²,width]
 → cat(class_embedding, x)  →  + positional_embedding
 → ln_pre → permute [L,N,E] → transformer → permute [N,L,E]
 → ln_post(CLS-токен) → proj (опционально)
```

Регистрируемые тензоры: `class_embedding`, `positional_embedding`, `proj` — в C++ порте
это `register_buffer` (в Python — `nn.Parameter`, см. roadmap).

### 4.4 CLIP (`src/RuCLIP.cpp:133-227`)

- `Visual` = `VisionTransformer`, `NVTransformer` = текстовый `Transformer` с каузальной
  маской `BuildAttentionMask()` (`fill(-inf) + triu_(1)`).
- `TokenEmbedding` (`nn::Embedding`), `PositionalEmbedding`, `LnFinal`, `TextProjection`,
  `LogitScale` (`log(1/0.07)`).
- `EncodeImage(pixel_values)` → `Visual(...)`.
- `EncodeText(input_ids)`:
  1. `TokenEmbedding(input_ids)` → `[N,77,W]`;
  2. `+ PositionalEmbedding`;
  3. `permute [L,N,E]` → `NVTransformer` → `permute [N,L,E]`;
  4. `LnFinal`;
  5. берётся эмбеддинг на позиции `eos_id` (`torch::where(input_ids == EosId)[1]`) и
     умножается на `TextProjection`.
- `forward(input_ids, pixel_values)`: L2-нормализация эмбеддингов, затем
  `logit_scale.exp() * image @ text.t()`. Возвращает **только** `logits_per_image`
  (в Python возвращается кортеж `(logits_per_image, logits_per_text)`).

---

## 5. Процессор (`RuCLIPProcessor`)

`src/RuCLIPProcessor.h`/`.cpp` — аналог `ruclip/processor.py`.

### 5.1 Токенизация (`EncodeText`, `PrepareTokens`)

1. Текст передаётся в `vkcom::BaseEncoder::encode_as_ids` (BPE, без `<BOS>`/`<EOS>`).
2. Обрезка до `TextSeqLength-2` (77−2=75) токенов, вставка `bos_id` в начало, `eos_id` в конец.
3. `PrepareTokens` дополняет последовательность нулями (`pad_id=0`) до длины `TextSeqLength`.

Идентификаторы по умолчанию: `pad_id=0`, `unk_id=1`, `bos_id=2`, `eos_id=3`.

> ⚠️ В отличие от Python, здесь **нет** `text.lower()` (см. roadmap).

### 5.2 Изображения (`EncodeImage`, `ResizeToInput`)

1. `ResizeToInput` масштабирует изображение с сохранением пропорций (letterbox) и
   дополняет чёрным до квадрата `ImageSize×ImageSize`.
2. `CVMatToTorchTensor` переводит `cv::Mat` (HWC, uint8) в тензор `[1,3,H,W]` float
   делением на 255.
3. `torch::data::transforms::Normalize<>(mean, std)` применяет нормализацию каналов.

> ⚠️ Python использует `RandomResizedCrop` (центральный квадрат + resize), а не letterbox
> с чёрными полями — это даёт расхождение эмбеддингов (см. roadmap).

### 5.3 Пакетный вызов `operator()`

- `operator()(texts, images)` → `(stack(тексты), pad_sequence(изображения).squeeze(0))`.
- `operator()(images)` — то же, но тексты берутся из кэша `CacheText(...)`.

---

## 6. Загрузка весов (`FromPretrained`)

`CLIP::FromPretrained(folder)` (`src/RuCLIP.h:158-194`):

1. Читает `folder/config.json` (размерности модели).
2. Создаёт `CLIP(...)` (в конструкторе запускается случайная инициализация параметров).
3. `torch::load(clip, folder/"jit_model.zip")` копирует веса из TorchScript-архива по
   **совпадающим именам** параметров/буферов.

`RuCLIPProcessor::FromPretrained(folder)` читает `config.json` (image_resolution,
context_length, mean/std) и `bpe.model`.

Экспорт `jit_model.zip` из PyTorch описан в `README.md`: `torch.jit.trace(clip, var)` +
`.save(...)`. Корректная загрузка целиком зависит от того, что имена модулей в C++ в
точности повторяют иерархию Python-модели (`visual.conv1.weight`,
`transformer.resblocks.0.attn.in_proj_weight` и т.д.).

---

## 7. Поток выполнения (`main.cpp`)

CLI-демо с двумя режимами (`--test_ind`):

### 7.1 `MatchImageAndText` (`test_ind=0`, `main.cpp:15-70`)

Классический zero-shot: для каждой картинки считаются логиты по всем текстовым меткам,
берётся softmax по меткам, выбирается максимальная.

```
processor(texts, images) → clip.forward → logits_per_image.softmax(-1)
```

### 7.2 `MatchOne2ManyImages` (`test_ind=1`, `main.cpp:73-163`)

Сравнение первого изображения со всеми остальными, текстовые метки используются как
«негативы»:

1. `embed = EncodeImage(images[0])` — референс.
2. `imgsFeatures = EncodeImage(images[1:])` — «позитивные» кандидаты.
3. `textFeatures = EncodeText(labels)` — «негативы».
4. `probs = Relevancy(embed, imgsFeatures, textFeatures)`.

> ⚠️ Этот режим содержит ошибки (см. roadmap: форма результата `Relevancy` не совпадает
> с индексацией `probs[row][col-1]`, а также краш при ≥2 «позитивах»).

---

## 8. Функция `Relevancy`

`inline torch::Tensor Relevancy(embeds, positives, negatives)` (`src/RuCLIPProcessor.h:112-149`).

Назначение (из комментария-примера в заголовке): даны эмбеддинги запроса, один
позитивный эмбеддинг и несколько негативных. Для каждого негатива вычисляется
`softmax(10·[sim_pos, sim_neg])`, находится «самый сложный» негатив (минимум вероятности
позитивного класса) и возвращается пара вероятностей `[pos_prob, neg_prob]`.

Текущая реализация **жёстко предполагает ровно один позитив**:

- `positive_vals = logits[..., 0:1]` берёт только первый столбец как позитив;
- `negative_vals = logits[..., 1:]` всё остальное считает негативами.

При `positives.size(0) > 1` это приводит к несоответствию форм в `torch::stack` и
исключению (детали и исправление — в `roadmap.md`).

---

## 9. Система сборки

- `cmake_minimum_required(3.8)`, C++17.
- `find_package(Torch REQUIRED)`, `find_package(OpenCV 4.6 REQUIRED)`.
- `include("CMakeLists.Files.txt")` добавляет исходники, заголовки и библиотеки.
- Для MSVC: post-build копирование DLL LibTorch в каталог exe, флаги `/bigobj /openmp`.
- Запуск через `.bat` (в `rel/Release/`) прописывает `PATH` к OpenCV/CUDA/cuDNN/TensorRT/Torch.

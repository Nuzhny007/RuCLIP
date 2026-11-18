# Найденные проблемы и план улучшений (roadmap)

Результат анализа C++-порта `RuCLIP` (корень: `C:\work\clip\ruclip\RuCLIP`).
Проблемы сгруппированы по severity. Ссылки даны в формате `файл:строка`.

---

## 1. Критические ошибки (runtime)

### 1.1 `Relevancy` падает при более чем одном позитиве
`src/RuCLIPProcessor.h:112-149`

> **Статус: ✅ исправлено.** Функция обобщена на произвольные `B`/`P`/`M` (результат
> `[B, P, 2]`), мёртвая `#if 0`-ветка удалена, вызывающий код в `main.cpp:136` обновлён.

Текущий код (до исправления) жёстко зашивал один позитив:

```cpp
auto positive_vals = logits.index({ "...", torch::indexing::Slice(0, 1) });          // только 1-й столбец
auto negative_vals = logits.index({ "...", torch::indexing::Slice(1, torch::indexing::None) });
```

При `positives.size(0) > 1`:
- `negative_vals` имеет форму `[B, P+M-1]`, а `repeated_pos` — `[B, M]`;
- `torch::stack({repeated_pos, negative_vals}, -1)` бросает исключение о несовпадении размеров.

Это ровно тот случай, который описан в задаче («подать несколько embeds»): в
`MatchOne2ManyImages` (`main.cpp:121`) вызов `Relevancy(embed, imgsFeatures, textFeatures)`
передаёт в качестве `positives` **несколько** других изображений.

> Причина: общий случай «несколько позитивов» не был реализован, а `#if 0`-ветка с
> попыткой обобщения отключена и тоже содержит ошибку.

**Исправление (применено)** — поддержка произвольного батча `B` и произвольного числа
позитивов `P`/негативов `M`, результат `[B, P, 2]`:

```cpp
inline torch::Tensor Relevancy(torch::Tensor embeds, torch::Tensor positives, torch::Tensor negatives)
{
    // embeds: [B, D], positives: [P, D], negatives: [M, D]  (уже L2-нормализованы)
    TORCH_CHECK(embeds.dim() == 2 && positives.dim() == 2 && negatives.dim() == 2,
                "expected 2D inputs");
    TORCH_CHECK(positives.size(0) >= 1 && negatives.size(0) >= 1,
                "need at least one positive and one negative");

    auto all  = torch::cat({ positives, negatives }, 0);          // [P+M, D]
    auto logits = torch::mm(embeds, all.t());                     // [B, P+M]

    auto pos_logits = logits.narrow(1, 0, positives.size(0));     // [B, P]
    auto neg_logits = logits.narrow(1, positives.size(0), negatives.size(0)); // [B, M]

    // пара «каждый позитив × каждый негатив»: [B, P, M, 2]
    auto sims = torch::stack({
        pos_logits.unsqueeze(2).expand({ -1, -1, negatives.size(0) }),
        neg_logits.unsqueeze(1).expand({ -1, positives.size(0), -1 })
    }, -1);

    auto smx = torch::softmax(10.0 * sims, -1);                   // [B, P, M, 2]
    auto pos_prob = smx.index({ "...", 0 });                      // [B, P, M]
    auto best_id = pos_prob.argmin(2, /*keepdim=*/true);          // [B, P, 1]  — самый сложный негатив
    auto result = torch::gather(smx, 2, best_id.unsqueeze(-1).expand({ -1, -1, -1, 2 }));
    return result.squeeze(2);                                     // [B, P, 2]
}
```

Для прежнего случая «один позитив» (`P==1`) форма становится `[B, 1, 2]`; при
необходимости совместимости добавьте `squeeze(1)` при `positives.size(0) == 1`.

### 1.2 `PrepareTokens` возвращает неопределённый тензор при `empty_positions == 0`
`src/RuCLIPProcessor.cpp:108-121`

```cpp
torch::Tensor result;                       // ← undefined
if (tokens.size() > TextSeqLength) { ... }  // ресайз до 77
int empty_positions = TextSeqLength - tokens.size();
if (empty_positions > 0)
    result = torch::cat({ ... });           // назначение только здесь
return result;                              // при empty_positions == 0 → undefined
```

Когда итоговая длина равна `TextSeqLength` (это происходит для текста с **≥ 75 BPE-токенов**,
т.к. `EncodeText` даёт `bos + 75 + eos = 77`), ветка `if` не выполняется и функция
возвращает undefined-тензор, что приводит к ошибке на дальнейших шагах
(`torch::stack`/`TokenEmbedding`). Для коротких меток (`кот`, `cat`) не проявляется —
поэтому демо «работает».

**Исправление** — обработать случай точного заполнения и упорядочить как в Python:

```cpp
torch::Tensor result = torch::tensor(tokens, torch::kLong);
int empty_positions = TextSeqLength - static_cast<int>(result.size(0));
if (empty_positions > 0)
    result = torch::cat({ result, torch::zeros(empty_positions, torch::kLong) });
else if (empty_positions < 0)
    result = result.narrow(0, 0, TextSeqLength);
return result;
```

### 1.3 `ResizeToInput`: `assert` на несуществующих переменных
`src/RuCLIPProcessor.cpp:87-88`

> **Статус: ✅ исправлено** (`m_XOffset`→`xOffset`, `m_YOffset`→`yOffset`, добавлен
> `#include <cassert>`). Без этого Debug-сборка (где `assert` активен) не компилируется.

```cpp
assert(2 * m_XOffset + newWidth == ImageSize);   // m_XOffset не объявлена
assert(2 * m_YOffset + newHeight == ImageSize);  // m_YOffset не объявлена
```

Должно быть `xOffset` / `yOffset` (строки 84-85). В Release-сборке `NDEBUG` вырезает
выражение, поэтому код компилируется и работает; в Debug — ошибка компиляции.
Заодно сами проверки стоит заменить на исключения (`TORCH_CHECK`/`throw`), а не `assert`.

### 1.4 `MatchOne2ManyImages`: индексация результата не соответствует форме
`src/main.cpp:121,133-148`

> **Статус: ⚠️ частично исправлено.** Индексация приведена к новой форме `[B, P, 2]`
> (`probs[row][col-1][0]`), выхода за границы больше нет; семантическая путаница
> («метки как негативы») остаётся.

`Relevancy` возвращает `[B, 2]` (позитивная/негативная вероятность), а цикл отображения
индексирует `probs[row][col-1]` для `col = 1..images.size()-1`:

```cpp
float prob = probs[row][col - 1].item<float>();
```

- при `images.size() > 3` — выход за границы тензора;
- даже без краша результат семантически неверен: метки здесь подаются как «негативы», а
  «позитивом» фактически считается только первый из остальных кадров.

Функцию нужно переписать согласованно: либо вычислять попарную близость
референс-картинки к каждой из остальных (без `Relevancy`), либо использовать исправленный
`Relevancy` с формой `[B, P, 2]` и корректно её разбирать.

---

## 2. Расхождения с референсной реализацией Python

### 2.1 Отсутствует `text.lower()`
`src/RuCLIPProcessor.cpp:41-62` (строка 47 — закомментированный `//text = lowercase(text);`)

Python (`ruclip/processor.py:32`) приводит текст к нижнему регистру перед токенизацией.
BPE-токенизатор этого не делает сам. Для заглавных букв C++-порт даст другую
токенизацию и, как следствие, другие (худшие) эмбеддинги. Нужно добавить корректный
lowercase с учётом UTF-8/локали (см. 4.6).

### 2.2 Препроцессинг изображений: letterbox вместо center-crop
`src/RuCLIPProcessor.cpp:65-97`

Python использует `T.RandomResizedCrop(image_size, scale=(1.,1.), ratio=(1.,1.))` —
центральный квадратный кроп + resize. C++ делает `ResizeToInput` — масштабирование с
сохранением пропорций и дополнением чёрными полями (letterbox). Это даёт систематическое
расхождение эмбеддингов с референсом. Рекомендуется реализовать center-crop+resize
(или хотя бы предоставить переключатель между двумя стратегиями).

### 2.3 Параметры зарегистрированы как буферы
`src/RuCLIP.cpp:101-103` (ViT: `class_embedding`, `positional_embedding`, `proj`) и
`src/RuCLIP.cpp:165-167` (`positional_embedding`, `text_projection`, `logit_scale`)

В Python это `nn.Parameter`, в C++ — `register_buffer`. Для инференса и загрузки весов
по именам это работает, но семантически неверно и хрупко:
- параметры не участвуют в `named_parameters()`/`requires_grad` (важно при дообучении);
- поведение `.to(dtype)`/`.train()`/`.eval()` для буферов и параметров различается.

Рекомендуется регистрировать их как `register_parameter` (через `torch::nn::Parameter`),
сохранив те же имена.

### 2.4 `CVMatToTorchTensor` без проверки непрерывности
`src/RuCLIPProcessor.cpp:4-12`

В базовой версии (`ruclip/DeliriumV01D/RuCLIP`) было:

```cpp
if (!img.isContinuous())
    img = img.clone();
```

Сейчас эта проверка убрана. Для непрерывного `cv::Mat` (результат `imread`, `ResizeToInput`)
работает, но для ROI/подматрицы `from_blob` прочитает память с неверными шагами →
мусор/краш. Вернуть проверку (или `img.clone()` перед `from_blob`).

---

## 3. Проблемы надёжности и качества кода

### 3.1 `FromPretrained` проглатывает ошибку загрузки весов
`src/RuCLIP.h:184-191`

```cpp
try { torch::load(clip, (folder / "jit_model.zip").string()); }
catch (std::exception& e) { std::cout << e.what() << std::endl; }
```

При любой ошибке (нет файла, несовпадение имён, повреждённый архив) модель молча
остаётся со **случайными** весами. Нужно пробрасывать исключение наружу (`throw`) или
возвращать статус ошибки, а не продолжать работу.

### 3.2 Нарушение правила пяти в `RuCLIPProcessor`
`src/RuCLIPProcessor.h:22,25-41`

Определены пользовательские move-assignment и деструктор (`= default`), из-за чего
компилятор не генерирует move-конструктор и копирующие операции (класс содержит
`std::unique_ptr`). Сейчас спасает только guaranteed copy elision (C++17) в
`FromPretrained`. Стоит явно объявить: `RuCLIPProcessor(RuCLIPProcessor&&) = default;`
и удалить копирование (`= delete`), либо написать полноценное правило пяти.

### 3.3 Абсолютный путь к Intel MKL в CMake
`CMakeLists.Files.txt:9-11`

```cmake
link_directories("C:/Program Files (x86)/Intel/oneAPI/mkl/2022.1.0/lib/intel64")
```

Машино-зависимый путь ломает сборку на других ПК. Заменить на
`find_package(MKL)`/`find_library(...)` или убрать (MKL уже тянется за LibTorch).

### 3.4 Чтение `config["mean"]`/`["std"]` без проверки наличия ключей
`src/RuCLIPProcessor.h:67-68`

Если `config.json` не содержит `mean`/`std`, `nlohmann::json` бросит исключение. Для
трёх моделей в `data/` ключи есть, но код должен быть устойчив: использовать
`config.contains(...)`/`value(...)` с дефолтом `{0.48145466, 0.4578275, 0.40821073}` и
`{0.26862954, 0.26130258, 0.27577711}`.

### 3.5 Хрупкая выборка позиции `eos_id`
`src/RuCLIP.cpp:204`

```cpp
torch::where(input_ids == EosId)[1]
```

Корректно только если в каждой строке ровно один `eos`. При прямом вызове `PrepareTokens`
с обрезкой (см. 1.2) или при ином формировании `input_ids` индексы «разъедутся».
Стоит добавить проверку числа `eos` на строку.

### 3.6 Отладочный вывод в боевом коде
`src/RuCLIP.h:162`, `src/RuCLIP.cpp:155,210,215`, `src/main.cpp` и др.

Разбросанные `std::cout` («transformer_width: …», «Relevancy: 0…», печать размеров)
засоряют вывод. Убрать или вынести за флаг `#ifdef _DEBUG`/параметр verbose.

### 3.7 Несамодостаточные include
- `src/RuCLIP.cpp` использует `std::numeric_limits<float>::infinity()` (нужен `<limits>`);
- `src/RuCLIP.h` использует `std::cout` (нужен `<iostream>`).

Сейчас компилируется транзитивно через LibTorch, но это неявная зависимость.

### 3.8 `from_blob` без `clone()` — завязка на время жизни `cv::Mat`
`src/RuCLIPProcessor.cpp:6`

Возвращаемый тензор после `.toType(kFloat).div(255)` — уже новая аллокация, поэтому в
текущем виде безопасно, но промежуточный тензор ссылается на память `cv::Mat`. Для
защиты от регрессий стоит явно `.clone()` результат `from_blob`.

### 3.9 `TorchTensorToCVMat`: `squeeze()` без аргумента
`src/RuCLIPProcessor.cpp:17`

`.squeeze()` убирает **все** размерности 1. Для `[1,1,H,W]` (grayscale) или батча >1
поведение неверно. Указать `squeeze(0)` и явно проверять число каналов.

### 3.10 Мёртвый код / недоделки
- `PrepareTokens` содержит ветку `if (tokens.size() > TextSeqLength)`, которая после
  обрезки в `EncodeText` фактически не срабатывает, но вносит путаницу (и см. 1.2).
- `//src/RuCLIPPredictor.{h,cpp}` в `CMakeLists.Files.txt` — недопортированный модуль.
- `dtype()`-аксессор в `CLIPImpl` закомментирован (`src/RuCLIP.h:141-144`).
- `register_buffer("attn_mask", ...)` закомментирован (`src/RuCLIP.cpp:20`).

---

## 4. Предложения по улучшению / новые возможности

### 4.1 Исправить `Relevancy` и переписать `MatchOne2ManyImages`
См. 1.1 и 1.4. После исправления формы результата `[B, P, 2]` привести цикл отображения
в соответствие (одна строка результата на каждую «остальную» картинку).

### 4.2 Возвращать полный кортеж из `forward`
Python возвращает `(logits_per_image, logits_per_text)`, C++ — только
`logits_per_image`. Добавить второй выход (или отдельный метод) для совместимости.

### 4.3 Полная поддержка fp16
`RCLayerNorm` уже умеет fp16, но `PositionalEmbedding`/`TextProjection` создаются через
`torch::empty` (float32) и нет единого `dtype()`-метода. Довести приведение типов до
полного соответствия `model.py` (`encode_image`/`encode_text` приводят вход к `self.dtype`).

### 4.4 `DecodeText` / обратная токенизация
В Python есть `decode_text` (пропуск `eos/bos/unk/pad`). Добавить в `RuCLIPProcessor`.

### 4.5 Порт `RuCLIPPredictor`
В Python `ruclip/predictor.py` реализует батчинг и шаблоны (`templates`). Полезно для
пакетной классификации; в CMake файлы уже зарезервированы.

### 4.6 Корректная обработка Unicode/локали
Помечено `//!!!Локали-юникоды`. Нижний регистр и работа со строками должны учитывать
UTF-8 (не `std::tolower` по одному байту). Рассмотреть ICU или
`std::wstring_convert`/UTF-8-aware lowercasing.

### 4.7 Тесты и CI
- Смоук-тест «текст↔картинка» с эталонными вероятностями из `README.md`
  (0.8879 / 0.0063 / 0.1058 и т.д.);
- тест `Relevancy` на формах `[1,·]` и `[B,·]`, `positives.size(0) ∈ {1, >1}`;
- сравнение эмбеддингов C++ vs Python на фиксированных входных данных.

### 4.8 Улучшить сборку
- Убрать жёсткие абсолютные пути (MKL, OpenCV `find_package(OpenCV 4.6)`), добавить
  `find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)`.
- Копировать DLL OpenCV в каталог exe (сейчас запуск только через `.bat` с `PATH`).
- Добавить `find_package(Threads REQUIRED)`.

---

## 5. Сводная таблица приоритетов

| # | Проблема | Severity | Файл |
|---|---|---|---|
| 1.1 | `Relevancy` крашится при >1 позитиве | ✅ Исправлено | `RuCLIPProcessor.h:114` |
| 1.2 | `PrepareTokens` → undefined при длинных текстах | Критично | `RuCLIPProcessor.cpp:108` |
| 1.3 | `assert(m_XOffset/...)` несуществующие переменные | ✅ Исправлено | `RuCLIPProcessor.cpp:87` |
| 1.4 | `MatchOne2ManyImages` OOB/семантика | ⚠️ Индексация исправлена | `main.cpp:121,133` |
| 2.1 | нет `lower()` | Высокий | `RuCLIPProcessor.cpp:41` |
| 2.2 | letterbox vs center-crop | Высокий | `RuCLIPProcessor.cpp:65` |
| 2.3 | параметры как буферы | Средний | `RuCLIP.cpp:101,165` |
| 2.4 | нет `isContinuous()` | Средний | `RuCLIPProcessor.cpp:4` |
| 3.1 | молчаливый провал загрузки весов | Высокий | `RuCLIP.h:184` |
| 3.2 | правило пяти | Средний | `RuCLIPProcessor.h:25` |
| 3.3 | абсолютный путь MKL | Средний | `CMakeLists.Files.txt:9` |
| 3.4 | нет проверки ключей config | Низкий | `RuCLIPProcessor.h:67` |
| 3.5–3.10 | индексация eos, debug-вывод, include, clone и т.д. | Низкий/Средний | — |

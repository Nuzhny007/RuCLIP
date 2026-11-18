#pragma once

#include "Backend/ClipTypes.h"

#include <vector>

namespace ruclip
{

/// Стандартный logit_scale CLIP/ruCLIP: exp(log(1 / 0.07)) = 1 / 0.07 ≈ 14.2857.
/// Новые бэкенды умеют читать значение из config.json; это — фолбэк.
constexpr float kDefaultLogitScale = 1.0f / 0.07f;

// ---------------------------------------------------------------------------
// Примитивы над TensorF32 (ни LibTorch, ни OpenCV, ни json).
// ---------------------------------------------------------------------------

/// [R, C] -> [C, R]
TensorF32 Transpose2D(const TensorF32& a);

/// [M, K] x [K, N] -> [M, N]
TensorF32 MatMul(const TensorF32& a, const TensorF32& b);

/// Построчная L2-нормализация: x[i, :] /= ||x[i, :]||_2.
/// Строки с нулевой нормой остаются нулевыми.
TensorF32 L2NormalizeRows(const TensorF32& a);

/// Построчный softmax по последней (2D: по столбцам) размерности.
/// temperature домножает логиты до softmax (для Relevancy используется 10).
TensorF32 SoftmaxRows(const TensorF32& a, float temperature = 1.0f);

/// Выбор строк по индексам: [N, ...] -> [rows.size(), ...].
TensorF32 GatherRows(const TensorF32& a, const std::vector<int32_t>& rows);

// ---------------------------------------------------------------------------
// Общий post-processing (одинаков для всех бэкендов).
// ---------------------------------------------------------------------------

/// logits_per_image = logitScale * (L2Norm(imageFeatures) @ L2Norm(textFeatures)^T)
/// imageFeatures [B, D], textFeatures [N, D] -> [B, N]
TensorF32 Logits(const TensorF32& imageFeatures, const TensorF32& textFeatures,
                 float logitScale = kDefaultLogitScale);

/// softmax(Logits(...)) по текстовым меткам: [B, N].
TensorF32 ProbsPerImage(const TensorF32& imageFeatures, const TensorF32& textFeatures,
                        float logitScale = kDefaultLogitScale);

/// Порт inline Relevancy из RuCLIPProcessor.h на нейтральные тензоры.
/// embeds [B, D], positives [P, D], negatives [M, D] -> [B, P, 2].
/// Для каждого (b, p) берётся «самый сложный» негатив (минимум P(positive)).
TensorF32 Relevancy(const TensorF32& embeds, const TensorF32& positives, const TensorF32& negatives);

// ---------------------------------------------------------------------------
// Выбор эмбеддинга EOS из полной последовательности text-энкодера.
// Позволяет не держать динамический Gather в ONNX-графе (риск для OpenCV DNN).
// ---------------------------------------------------------------------------

/// fullSequence [B, L, D], eosPos [B] -> [B, D]
TensorF32 GatherEos(const TensorF32& fullSequence, const std::vector<int32_t>& eosPos);

/// Плоский вариант: full [B * L * D] -> out [B * D].
std::vector<float> GatherEos(const std::vector<float>& full, int B, int L, int D,
                             const std::vector<int32_t>& eosPos);

} // namespace ruclip

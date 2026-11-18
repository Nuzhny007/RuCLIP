#pragma once

#include "Backend/ClipTypes.h"

#include <memory>

namespace ruclip
{

/// Бэкенд-независимый интерфейс инференса ruCLIP.
///
/// Контракт (см. docs/backend-architecture.md, раздел 3):
///  - EncodeImage: ImageBatch [B,3,H,W] (нормализован) -> TensorF32 [B, D]
///  - EncodeText : TextBatch  [B,L] i32               -> TensorF32 [B, D]
///
/// Выходные эмбеддинги НЕ L2-нормализованы: L2-норма, logit_scale, softmax и
/// Relevancy выполняются общим кодом (Backend/ClipMath.h) одинаково для всех бэкендов.
class IClipBackend
{
public:
	virtual ~IClipBackend() = default;

	virtual BackendKind kind() const = 0;

	/// Эффективный множитель логитов: exp(logit_scale).
	/// Torch-бэкенд берёт значение из весов модели; ONNX-бэкенды — из config.json.
	virtual float logitScale() const = 0;

	virtual TensorF32 EncodeImage(const ImageBatch& images) = 0;
	virtual TensorF32 EncodeText(const TextBatch& text) = 0;
};

/// Создаёт бэкенд указанного типа или бросает std::runtime_error,
/// если он не собран / ещё не реализован.
std::unique_ptr<IClipBackend> CreateBackend(BackendKind kind, const BackendConfig& cfg);

} // namespace ruclip

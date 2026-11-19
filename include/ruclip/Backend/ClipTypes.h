#pragma once

#include <cstdint>
#include <filesystem>
#include <utility>
#include <vector>

namespace ruclip
{

/// Минимальный CPU-тензор float32 (row-major), общий для всех бэкендов.
///
/// Форма произвольная: 2D достаточно для эмбеддингов и логитов, а трёхмерная
/// [B, P, 2] нужна для Relevancy. Для 2D-случая есть удобные rows()/cols().
struct TensorF32
{
	std::vector<float> data;
	std::vector<int64_t> shape;

	TensorF32() = default;
	explicit TensorF32(std::vector<int64_t> s) : shape(std::move(s)) { data.assign(Numel(), 0.f); }

	int64_t Numel() const
	{
		int64_t n = 1;
		for (auto d : shape)
			n *= d;
		return n;
	}

	int Dim() const { return static_cast<int>(shape.size()); }

	int64_t Size(int i) const
	{
		return (i >= 0 && i < Dim()) ? shape[static_cast<size_t>(i)] : 0;
	}

	int64_t rows() const { return Size(0); }
	int64_t cols() const { return Size(1); }

	bool Empty() const { return data.empty(); }

	float* Data() { return data.data(); }
	const float* Data() const { return data.data(); }

	/// Доступ по плоскому индексу (row-major).
	float& Flat(int64_t i) { return data[static_cast<size_t>(i)]; }
	float Flat(int64_t i) const { return data[static_cast<size_t>(i)]; }

	/// Доступ к элементу 2D-тензора.
	float& At(int64_t i, int64_t j) { return data[static_cast<size_t>(i * cols() + j)]; }
	const float& At(int64_t i, int64_t j) const { return data[static_cast<size_t>(i * cols() + j)]; }
};

/// Токены: [B * L] int32 + позиция EOS для каждой строки.
/// int32, а не int64: токены <= 49408, и это снимает проблему int64-Gather
/// у OpenCV DNN.
struct TextBatch
{
	std::vector<int32_t> ids;      // [B * L], порядок bos .. eos .. pad(0)
	std::vector<int32_t> eos_pos;  // [B], индекс EOS внутри L=77
	int B = 0;
	int L = 0;
};

/// Изображения: [B * 3 * H * W] float32, уже (HWC->CHW, /255, normalize mean/std).
struct ImageBatch
{
	std::vector<float> pixels;     // [B * C * H * W]
	int B = 0, C = 0, H = 0, W = 0;
};

/// Идентификатор бэкенда (ключи --backend=torch|trt|dnn).
enum class BackendKind
{
	Torch,
	TensorRT,
	Dnn,
};

/// Общая конфигурация бэкенда (заполняется в CLI/фабрике).
///
/// TensorRT-бэкенд кэширует engine рядом с ONNX:
///   <model>/clip_visual.<fp32|fp16>.gpu<N>.b<B>.engine
///   <model>/clip_textual.<fp32|fp16>.gpu<N>.b<B>.engine
/// N — фактический индекс CUDA-устройства, B — maxBatch. Кэш переиспользуется,
/// а при несовместимости с текущей версией TensorRT/GPU пересобирается.
struct BackendConfig
{
	std::filesystem::path modelDir;	///< каталог модели (config.json, bpe.model, ...)
	int deviceIndex = 0;			///< индекс CUDA-устройства (TensorRT); входит в имя engine (gpu<N>)
	int maxBatch = 8;			///< TensorRT: верхняя граница batch в optimization profile; входит в имя engine (b<B>)
	bool useFp16 = false;			///< TensorRT: построить FP16-engine; входит в имя engine (fp32/fp16)
	bool verbose = false;
};

} // namespace ruclip

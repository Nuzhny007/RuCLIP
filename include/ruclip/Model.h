#pragma once

#include "ruclip/Backend/ClipTypes.h"
#include "ruclip/Backend/IClipBackend.h"
#include "ruclip/RuCLIPProcessor.h"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace ruclip
{

/// Глобальный enum из RuCLIPProcessor.h доступен и как ruclip::ImagePreprocess.
using ::ImagePreprocess;

/// Параметры загрузки модели для ruclip::Model.
struct ModelOptions
{
	std::filesystem::path modelDir;			///< каталог модели (config.json, bpe.model, ...)
	BackendKind backend = BackendKind::Torch;	///< torch | trt | dnn
	int deviceIndex = 0;				///< индекс CUDA-устройства (TensorRT / LibTorch CUDA)
	int maxBatch = 8;				///< TensorRT: верхняя граница batch в optimization profile
	bool useFp16 = false;				///< TensorRT / LibTorch: FP16
	bool verbose = false;				///< подробный лог бэкенда
	ImagePreprocess preprocess = ImagePreprocess::Letterbox;
};

/// Высокоуровневый фасад библиотеки: процессор + бэкенд инференса.
///
/// Повторяет оркестрацию, которая раньше жила в CLI, и избавляет стороннее
/// приложение от ручной сборки RuCLIPProcessor + CreateBackend:
///
/// @code
///   ruclip::ModelOptions opt;
///   opt.modelDir = "C:/models/ruclip-vit-large-patch14-336";
///   opt.backend  = ruclip::BackendKind::TensorRT;
///   opt.useFp16  = true;
///   ruclip::Model model(opt);
///   auto probs = model.ProbsPerImage({image}, {"кот", "медведь", "лиса"});
/// @endcode
class Model
{
public:
	explicit Model(const ModelOptions& options);

	Model(Model&&) = default;
	Model& operator=(Model&&) = default;
	Model(const Model&) = delete;
	Model& operator=(const Model&) = delete;

	BackendKind backendKind() const { return backend_->kind(); }
	float logitScale() const { return backend_->logitScale(); }

	const RuCLIPProcessor& processor() const { return processor_; }
	IClipBackend& backend() { return *backend_; }
	const IClipBackend& backend() const { return *backend_; }

	/// Один кадр -> эмбеддинг [1, D] (без L2-нормы).
	TensorF32 EncodeImage(const cv::Mat& image);
	/// Батч кадров -> эмбеддинги [B, D] (без L2-нормы).
	TensorF32 EncodeImage(const std::vector<cv::Mat>& images);

	/// Один текст -> эмбеддинг [1, D] (без L2-нормы).
	TensorF32 EncodeText(const std::string& text);
	/// Батч текстов -> эмбеддинги [B, D] (без L2-нормы).
	TensorF32 EncodeText(const std::vector<std::string>& texts);

	/// softmax(logitScale * norm(I) @ norm(T)^T): [B изображений, N текстов].
	TensorF32 ProbsPerImage(const std::vector<cv::Mat>& images,
	                        const std::vector<std::string>& texts);

	/// Relevancy: [B, P, 2] — см. ruclip::Relevancy в Backend/ClipMath.h.
	/// reference должен содержать ровно один кадр.
	TensorF32 Relevancy(const std::vector<cv::Mat>& reference,
	                    const std::vector<cv::Mat>& candidates,
	                    const std::vector<std::string>& texts);

private:
	RuCLIPProcessor processor_;
	std::unique_ptr<IClipBackend> backend_;
};

} // namespace ruclip

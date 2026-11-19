#pragma once

#include "ruclip/Backend/ClipTypes.h"

#include "ruclip/json.hpp"
#include "ruclip/youtokentome/bpe.h"

#include <opencv2/imgproc/imgproc.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

/// Стратегия приведения изображения к квадрату ImageSize×ImageSize.
enum class ImagePreprocess
{
	Letterbox,	// масштаб с сохранением пропорций + чёрные поля (по умолчанию)
	CenterCrop,	// центральный квадратный кроп + resize (как RandomResizedCrop в Python)
	Stretch,	// прямая растяжка к квадрату (как в исходном примере README)
};

/// Процессор ruCLIP: токенизация текста и препроцессинг изображений.
///
/// Бэкенд-независим: возвращает нейтральные TextBatch / ImageBatch
/// (см. Backend/ClipTypes.h), а не torch::Tensor.
class RuCLIPProcessor
{
public:
	RuCLIPProcessor() = default;

	RuCLIPProcessor(const std::string& tokenizer_path,
		const int image_size = 224,
		const int text_seq_length = 77,
		const std::vector<double> norm_mean = { 0.48145466, 0.4578275, 0.40821073 },
		const std::vector<double> norm_std = { 0.26862954, 0.26130258, 0.27577711 },
		const ImagePreprocess preprocess = ImagePreprocess::Letterbox);
	~RuCLIPProcessor() = default;

	RuCLIPProcessor(const RuCLIPProcessor&) = delete;
	RuCLIPProcessor& operator=(const RuCLIPProcessor&) = delete;
	RuCLIPProcessor(RuCLIPProcessor&&) = default;
	RuCLIPProcessor& operator=(RuCLIPProcessor&&) = default;

	/// Токенизация одного текста -> TextBatch (B=1, L=TextSeqLength).
	ruclip::TextBatch EncodeText(const std::string& text) const;
	/// Батч текстов -> TextBatch (B=texts.size()).
	ruclip::TextBatch PrepareTexts(const std::vector<std::string>& texts) const;

	/// Препроцессинг одного изображения -> ImageBatch (B=1, C=3, H=W=ImageSize).
	ruclip::ImageBatch EncodeImage(const cv::Mat& img) const;
	/// Батч изображений -> ImageBatch (B=images.size()).
	ruclip::ImageBatch PrepareImages(const std::vector<cv::Mat>& images) const;

	std::pair<ruclip::TextBatch, ruclip::ImageBatch> operator()(
		const std::vector<std::string>& texts, const std::vector<cv::Mat>& images) const;
	std::pair<ruclip::TextBatch, ruclip::ImageBatch> operator()(
		const std::vector<cv::Mat>& images) const;

	void CacheText(const std::vector<std::string>& texts);
	const std::vector<ruclip::TextBatch>& GetCachedTexts() const;

	///
	int GetImageSize() const noexcept
	{
		return ImageSize;
	}

	///
	static RuCLIPProcessor FromPretrained(const std::filesystem::path& folder,
		const ImagePreprocess preprocess = ImagePreprocess::Letterbox)
	{
		std::filesystem::path tokenizer_path = folder / "bpe.model";
		using json = nlohmann::json;
		std::ifstream f(folder / "config.json");
		json config = json::parse(f);

		// Ключи mean/std могут отсутствовать — используем значения по умолчанию ru-clip.
		const std::vector<double> default_mean = { 0.48145466, 0.4578275, 0.40821073 };
		const std::vector<double> default_std  = { 0.26862954, 0.26130258, 0.27577711 };
		auto mean = config.value("mean", default_mean);
		auto std  = config.value("std", default_std);

		return RuCLIPProcessor(tokenizer_path.string(),
                               int(config["image_resolution"]),
                               int(config["context_length"]),
                               mean, std,
                               preprocess);
	}

private:
	uint32_t eos_id = 3;
	uint32_t bos_id = 2;
	uint32_t unk_id = 1;
	uint32_t pad_id = 0;
	int ImageSize{ 224 };
	int TextSeqLength{ 77 };
	std::vector<double> NormMean;
	std::vector<double> NormStd;
	ImagePreprocess Preprocess{ ImagePreprocess::Letterbox };
	std::unique_ptr<vkcom::BaseEncoder> Tokenizer;

	/// Закэшированные тексты (для operator()(images) без явных меток).
	std::vector<ruclip::TextBatch> m_cachedTexts;

	/// Приведение длины токенов к TextSeqLength + вычисление позиции EOS.
	ruclip::TextBatch PrepareTokens(std::vector<int32_t> tokens) const;

	/// Приведение изображения к ImageSize×ImageSize.
	cv::Mat ResizeToInput(const cv::Mat& img) const;
};

#pragma once

#include "json.hpp"
#include "bpe.h"
#include "TorchHeader.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <filesystem>
#include <fstream>

/// Стратегия приведения изображения к квадрату ImageSize×ImageSize.
enum class ImagePreprocess
{
	Letterbox,	// масштаб с сохранением пропорций + чёрные поля (текущее поведение, по умолчанию)
	CenterCrop,	// центральный квадратный кроп + resize (как RandomResizedCrop в Python)
};

///
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

	///
	RuCLIPProcessor& operator=(RuCLIPProcessor&& processor) noexcept
	{
		eos_id = processor.eos_id;
		bos_id = processor.bos_id;
		unk_id = processor.unk_id;
		pad_id = processor.pad_id;
		ImageSize = processor.ImageSize;
		TextSeqLength = processor.TextSeqLength;
		NormMean = processor.NormMean;
		NormStd = processor.NormStd;
		Preprocess = processor.Preprocess;
		
		Tokenizer = std::move(processor.Tokenizer);

		m_textsTensors = processor.m_textsTensors;

		return *this;
	}

	///!!!Локали-юникоды
	torch::Tensor EncodeText(const /*std::vector<*/std::string &text) const;
	torch::Tensor PrepareTokens(/*std::vector<*/std::vector<int32_t> tokens) const;	//Передаю по значению чтобы внутри иметь дело с копией
	torch::Tensor EncodeImage(const cv::Mat& img) const;
	std::pair<torch::Tensor, torch::Tensor> operator()(const std::vector <std::string>& texts, const std::vector <cv::Mat>& images) const;
	std::pair<torch::Tensor, torch::Tensor> operator()(const std::vector <cv::Mat>& images) const;

	void CacheText(const std::vector <std::string>& texts);

	///
	int GetImageSize() const noexcept
	{
		return ImageSize;
	}

	///
	static RuCLIPProcessor FromPretrained(const std::filesystem::path &folder,
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

	const std::vector<torch::Tensor>& GetTextTensors() const;

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

	std::vector<torch::Tensor> m_textsTensors;

	///
	cv::Mat ResizeToInput(const cv::Mat& img) const;
};

///relevancy: для каждого запроса ищет самый «сложный» негатив.
/// Вход: embeds [B, D], positives [P, D], negatives [M, D] (L2-нормализованы).
/// Выход: [B, P, 2] — (P(positive), P(negative)) для каждого позитива.
///
///std::vector<torch::Tensor> canon_texts_tensors;
///canon_texts_tensors.push_back(ClipProcessor->EncodeText(std::string("объект")));
///canon_texts_tensors.push_back(ClipProcessor->EncodeText(std::string("вещи")));
///canon_texts_tensors.push_back(ClipProcessor->EncodeText(std::string("текстура")));
///int negatives_len =  (int)canon_texts_tensors.size();
///auto canon_features = Clip->EncodeText(torch::stack(canon_texts_tensors).to(Device)).to(torch::kCPU); ///[3, 768]
///canon_features = canon_features / canon_features.norm(2/*L2*/, -1, true);
///auto input = ClipProcessor->EncodeText(std::string("малый барабан"));
///auto text_features = Clip->EncodeText(input.unsqueeze(0).to(Device)).to(torch::kCPU);		///[1, 768]
///text_features = text_features / text_features.norm(2/*L2*/, -1, true);
///torch::Tensor image_features = PyramidClipEmbedding.GetPixelValue(i,j,0.5f,img_id,pyramid_embedder_properties,cv::Size(data.W, data.H)).to(torch::kCPU);
///image_features = image_features / image_features.norm(2/*L2*/, -1, true);
///torch::Tensor rel = Relevancy(image_features, text_features, canon_features);
///float lv = rel.index({0,0,0}).item<float>(); // вероятность позитивного класса для 0-го позитива

inline torch::Tensor Relevancy(torch::Tensor embeds, torch::Tensor positives, torch::Tensor negatives)
{
	// embeds    : [B, D] — эмбеддинги запроса (должны быть L2-нормализованы)
	// positives : [P, D] — позитивные эмбеддинги
	// negatives : [M, D] — негативные эмбеддинги
	// Результат : [B, P, 2] — для каждого позитива пара (P(positive), P(negative)),
	//             посчитанная против его самого «сложного» негатива.
	TORCH_CHECK(embeds.dim() == 2 && positives.dim() == 2 && negatives.dim() == 2,
		"Relevancy: expected 2D inputs [N, D]");
	TORCH_CHECK(positives.size(0) >= 1, "Relevancy: at least one positive is required");
	TORCH_CHECK(negatives.size(0) >= 1, "Relevancy: at least one negative is required");

	const int64_t positives_len = positives.size(0);
	const int64_t negatives_len = negatives.size(0);

	auto embeds2 = torch::cat({ positives, negatives }, 0);					// [P + M, D]
	auto logits = torch::mm(embeds, embeds2.t());							// [B, P + M]
	auto positive_vals = logits.narrow(1, 0, positives_len);				// [B, P]
	auto negative_vals = logits.narrow(1, positives_len, negatives_len);	// [B, M]

	//Пары «каждый позитив x каждый негатив»: [B, P, M, 2]
	auto sims = torch::stack({
			positive_vals.unsqueeze(2).expand({ -1, -1, negatives_len }),
			negative_vals.unsqueeze(1).expand({ -1, positives_len, -1 })
		}, -1);

	auto smx = torch::softmax(10.0 * sims, -1);								// [B, P, M, 2]
	//Для каждого позитива находим индекс самого сложного негатива (минимум вероятности позитивного класса)
	auto best_id = smx.index({ "...", 0 }).argmin(2, /*keepdim=*/true);	// [B, P, 1]
	//Собираем результаты для выбранных негативов
	auto result = torch::gather(smx, 2, best_id.unsqueeze(-1).expand({ -1, -1, -1, 2 }));	// [B, P, 1, 2]
	return result.squeeze(2);												// [B, P, 2]
}

#include "RuCLIPProcessor.h"
#include "TextLower.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace
{

/// cv::Mat (8U, HxWx3, BGR как в cv::imread) -> нормализованный float [C, H, W].
///
/// Каналы и mean/std применяются по индексу канала — в точности как раньше
/// (CVMatToTorchTensor с permute(2,0,1) + Normalize(mean, std)).
std::vector<float> MatToNormalizedCHW(const cv::Mat& img,
                                      const std::vector<double>& mean,
                                      const std::vector<double>& std_)
{
	CV_Assert(!img.empty());
	CV_Assert(img.channels() == 3);
	CV_Assert(mean.size() == 3 && std_.size() == 3);

	const int H = img.rows;
	const int W = img.cols;
	std::vector<float> out(static_cast<size_t>(3) * static_cast<size_t>(H) * static_cast<size_t>(W));

	for (int y = 0; y < H; ++y)
	{
		const uchar* row = img.ptr<uchar>(y);
		for (int x = 0; x < W; ++x)
		{
			for (int c = 0; c < 3; ++c)
			{
				const float v = static_cast<float>(row[x * 3 + c]) / 255.0f;
				out[(static_cast<size_t>(c) * H + y) * W + x] =
					(v - static_cast<float>(mean[c])) / static_cast<float>(std_[c]);
			}
		}
	}
	return out;
}

/// Слияние батчей B=1 в один батч B=N.
ruclip::TextBatch MergeTextBatches(const std::vector<ruclip::TextBatch>& singles)
{
	ruclip::TextBatch out;
	out.B = static_cast<int>(singles.size());
	if (singles.empty())
		return out;

	out.L = singles.front().L;
	out.ids.reserve(static_cast<size_t>(out.B) * static_cast<size_t>(out.L));
	out.eos_pos.reserve(singles.size());
	for (const auto& s : singles)
	{
		if (s.L != out.L)
			throw std::runtime_error("MergeTextBatches: inconsistent sequence length");
		out.ids.insert(out.ids.end(), s.ids.begin(), s.ids.end());
		out.eos_pos.push_back(s.eos_pos.empty() ? 0 : s.eos_pos.front());
	}
	return out;
}

} // namespace

///
RuCLIPProcessor::RuCLIPProcessor(
	const std::string& tokenizer_path,
	const int image_size /*= 224*/,
	const int text_seq_length /*= 77*/,
	const std::vector<double> norm_mean /*= { 0.48145466, 0.4578275, 0.40821073 }*/,
	const std::vector<double> norm_std /*= { 0.26862954, 0.26130258, 0.27577711 }*/,
	const ImagePreprocess preprocess /*= ImagePreprocess::Letterbox*/
) : ImageSize(image_size), TextSeqLength(text_seq_length), NormMean(norm_mean), NormStd(norm_std), Preprocess(preprocess)
{
	vkcom::Status status;
	Tokenizer = std::make_unique<vkcom::BaseEncoder>(tokenizer_path, -1, &status);
}

///!!!Локали-юникоды
ruclip::TextBatch RuCLIPProcessor::EncodeText(const std::string& text) const
{
	std::vector<std::vector<int32_t>> ret_ids;
	vkcom::Status status;
	// Как в оригинальном ru-clip: text = text.lower() перед токенизацией.
	// Токенизатор YouTokenToMe регистр сам не понижает, поэтому делаем это здесь.
	const std::string lowered = ToLowerUtf8(text);
	//output_type = vkcom::OutputType::ID, bos = false, eos = false, reverse = false, dropout_prob = 0.0
	std::vector<std::string> texts{ lowered };
	status = Tokenizer->encode_as_ids(texts, &ret_ids);
	if (status.code != 0)
		throw std::runtime_error("RuCLIPProcessor::EncodeText error : " + status.message);

	std::vector<int32_t> it = ret_ids[0];
	if (it.size() > static_cast<size_t>(TextSeqLength - 2))
		it.resize(TextSeqLength - 2);
	it.insert(it.begin(), static_cast<int32_t>(bos_id));	// vector сдвинет при вставке
	it.push_back(static_cast<int32_t>(eos_id));

	return PrepareTokens(std::move(it));
}

///
ruclip::TextBatch RuCLIPProcessor::PrepareTokens(std::vector<int32_t> tokens) const
{
	// Приводим длину ровно к TextSeqLength, как в оригинальном ru-clip:
	//  - короче  -> дополняем pad_id (0);
	//  - длиннее -> обрезаем, сохраняя последний токен (обычно eos).
	const int empty_positions = TextSeqLength - static_cast<int>(tokens.size());
	if (empty_positions > 0)
	{
		tokens.resize(TextSeqLength, static_cast<int32_t>(pad_id));
	}
	else if (empty_positions < 0)
	{
		const int32_t back = tokens.back();
		tokens.resize(TextSeqLength);
		tokens.back() = back;
	}

	ruclip::TextBatch batch;
	batch.B = 1;
	batch.L = TextSeqLength;
	batch.ids = std::move(tokens);

	// Позиция EOS: первое вхождение eos_id (в EncodeText он вставляется один раз).
	int32_t eosPos = static_cast<int32_t>(batch.ids.size()) - 1;
	for (size_t i = 0; i < batch.ids.size(); ++i)
	{
		if (batch.ids[i] == static_cast<int32_t>(eos_id))
		{
			eosPos = static_cast<int32_t>(i);
			break;
		}
	}
	batch.eos_pos.push_back(eosPos);
	return batch;
}

///
ruclip::TextBatch RuCLIPProcessor::PrepareTexts(const std::vector<std::string>& texts) const
{
	std::vector<ruclip::TextBatch> singles;
	singles.reserve(texts.size());
	for (const auto& t : texts)
		singles.push_back(EncodeText(t));
	return MergeTextBatches(singles);
}

///
cv::Mat RuCLIPProcessor::ResizeToInput(const cv::Mat& img) const
{
	CV_Assert(!img.empty());

	cv::Mat newImg(cv::Size(ImageSize, ImageSize), img.type(), cv::Scalar(0, 0, 0));

	if (Preprocess == ImagePreprocess::CenterCrop)
	{
		// Python-совместимый вариант: как RandomResizedCrop(scale=(1,1), ratio=(1,1))
		// при инференсе — центральный квадрат min(H, W) × min(H, W), затем resize.
		// torchvision по умолчанию использует билинейную интерполяцию.
		const int side = std::min(img.rows, img.cols);
		const int x = (img.cols - side) / 2;
		const int y = (img.rows - side) / 2;
		cv::Mat crop = img(cv::Rect(x, y, side, side));
		cv::resize(crop, newImg, cv::Size(ImageSize, ImageSize), 0, 0, cv::INTER_LINEAR);
		return newImg;
	}

	// Letterbox (по умолчанию): масштаб с сохранением пропорций + чёрные поля.
	const float r = std::min(static_cast<float>(ImageSize) / static_cast<float>(img.rows),
	                         static_cast<float>(ImageSize) / static_cast<float>(img.cols));
	int newHeight = cvRound(img.rows * r);
	int newWidth = cvRound(img.cols * r);

	// Additional checks for images with non even dims
	if ((ImageSize - newWidth) % 2)
		newWidth--;
	if ((ImageSize - newHeight) % 2)
		newHeight--;

	// Заменяем assert (который исчезает под NDEBUG) на проверки, активные
	// в любой конфигурации сборки: нарушение инварианта не должно проходить
	// незамеченным в Release.
	CV_Assert((ImageSize - newWidth) % 2 == 0 && (ImageSize - newHeight) % 2 == 0);

	const int xOffset = (ImageSize - newWidth) / 2;
	const int yOffset = (ImageSize - newHeight) / 2;

	CV_Assert(2 * xOffset + newWidth == ImageSize &&
	          2 * yOffset + newHeight == ImageSize);

	cv::resize(img, newImg(cv::Rect(xOffset, yOffset, newWidth, newHeight)), cv::Size(newWidth, newHeight), 0, 0, cv::INTER_CUBIC);
	return newImg;
}

///
ruclip::ImageBatch RuCLIPProcessor::EncodeImage(const cv::Mat& img) const
{
	const cv::Mat resized = ResizeToInput(img);

	ruclip::ImageBatch batch;
	batch.B = 1;
	batch.C = resized.channels();
	batch.H = resized.rows;
	batch.W = resized.cols;
	batch.pixels = MatToNormalizedCHW(resized, NormMean, NormStd);
	return batch;
}

///
ruclip::ImageBatch RuCLIPProcessor::PrepareImages(const std::vector<cv::Mat>& images) const
{
	std::vector<ruclip::ImageBatch> singles;
	singles.reserve(images.size());
	for (const auto& img : images)
		singles.push_back(EncodeImage(img));

	ruclip::ImageBatch out;
	out.B = static_cast<int>(singles.size());
	if (singles.empty())
		return out;

	out.C = singles.front().C;
	out.H = singles.front().H;
	out.W = singles.front().W;

	const size_t one = static_cast<size_t>(out.C) * static_cast<size_t>(out.H) * static_cast<size_t>(out.W);
	out.pixels.reserve(static_cast<size_t>(out.B) * one);
	for (const auto& s : singles)
	{
		if (s.C != out.C || s.H != out.H || s.W != out.W)
			throw std::runtime_error("PrepareImages: inconsistent image sizes");
		out.pixels.insert(out.pixels.end(), s.pixels.begin(), s.pixels.end());
	}
	return out;
}

///
void RuCLIPProcessor::CacheText(const std::vector<std::string>& texts)
{
	m_cachedTexts.clear();
	for (const auto& t : texts)
		m_cachedTexts.push_back(EncodeText(t));
}

///
const std::vector<ruclip::TextBatch>& RuCLIPProcessor::GetCachedTexts() const
{
	return m_cachedTexts;
}

///
std::pair<ruclip::TextBatch, ruclip::ImageBatch> RuCLIPProcessor::operator()(
	const std::vector<std::string>& texts, const std::vector<cv::Mat>& images) const
{
	return { PrepareTexts(texts), PrepareImages(images) };
}

///
std::pair<ruclip::TextBatch, ruclip::ImageBatch> RuCLIPProcessor::operator()(
	const std::vector<cv::Mat>& images) const
{
	return { MergeTextBatches(m_cachedTexts), PrepareImages(images) };
}

#include "ruclip/Backend/DnnBackend.h"

#include "ruclip/Backend/ClipMath.h"
#include "ruclip/json.hpp"

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

namespace ruclip
{
namespace
{

/// cv::Mat (float32) -> TensorF32 с сохранением формы.
TensorF32 MatToTensorF32(const cv::Mat& m)
{
	CV_Assert(m.type() == CV_32F);
	const cv::Mat c = m.isContinuous() ? m : m.clone();

	TensorF32 out;
	out.shape.assign(c.size.p, c.size.p + c.dims);
	const float* p = reinterpret_cast<const float*>(c.data);
	out.data.assign(p, p + c.total());
	return out;
}

class DnnBackend final : public IClipBackend
{
public:
	explicit DnnBackend(const BackendConfig& cfg)
	{
		using json = nlohmann::json;
		const std::filesystem::path configPath = cfg.modelDir / "config.json";
		std::ifstream f(configPath);
		if (!f)
			throw std::runtime_error("DNN: cannot open " + configPath.string());
		json config = json::parse(f);

		imageSize_ = int(config["image_resolution"]);
		contextLength_ = int(config["context_length"]);
		logitScale_ = config.contains("logit_scale")
			? float(config["logit_scale"].get<double>())
			: kDefaultLogitScale;

		const std::string visualOnnx = (cfg.modelDir / "clip_visual.onnx").string();
		const std::string textualOnnx = (cfg.modelDir / "clip_textual.onnx").string();
		if (!std::filesystem::exists(visualOnnx))
			throw std::runtime_error("DNN: missing ONNX model " + visualOnnx);
		if (!std::filesystem::exists(textualOnnx))
			throw std::runtime_error("DNN: missing ONNX model " + textualOnnx);

		// Новый graph-движок OpenCV 5 (ENGINE_OPENCV), а не классический.
		visualNet_ = cv::dnn::readNetFromONNX(visualOnnx, cv::dnn::ENGINE_OPENCV);
		textualNet_ = cv::dnn::readNetFromONNX(textualOnnx, cv::dnn::ENGINE_OPENCV);
	}

	BackendKind kind() const override { return BackendKind::Dnn; }

	float logitScale() const override { return logitScale_; }

	TensorF32 EncodeImage(const ImageBatch& images) override
	{
		if (images.B <= 0)
			throw std::runtime_error("DnnBackend::EncodeImage: empty batch");
		if (images.C != 3 || images.H != imageSize_ || images.W != imageSize_)
			throw std::runtime_error("DnnBackend::EncodeImage: unexpected image shape");

		int sizes[4] = { images.B, images.C, images.H, images.W };
		// Обёртка над внешней памятью; setInput() делает copyTo().
		cv::Mat blob(4, sizes, CV_32F, const_cast<float*>(images.pixels.data()));
		visualNet_.setInput(blob, "pixel_values");

		const cv::Mat out = visualNet_.forward("image_features");	// [B, D]
		return MatToTensorF32(out);
	}

	TensorF32 EncodeText(const TextBatch& text) override
	{
		if (text.B <= 0)
			throw std::runtime_error("DnnBackend::EncodeText: empty batch");
		if (text.L != contextLength_)
			throw std::runtime_error("DnnBackend::EncodeText: unexpected sequence length");

		// input_ids в ONNX экспортирован как int32 (CV_32S).
		cv::Mat blob(text.B, text.L, CV_32S, const_cast<int32_t*>(text.ids.data()));
		textualNet_.setInput(blob, "input_ids");

		const cv::Mat out = textualNet_.forward("text_features");	// [B, L, D]
		const TensorF32 full = MatToTensorF32(out);
		return GatherEos(full, text.eos_pos);
	}

private:
	int imageSize_ = 0;
	int contextLength_ = 0;
	float logitScale_ = kDefaultLogitScale;
	cv::dnn::Net visualNet_;
	cv::dnn::Net textualNet_;
};

} // namespace

std::unique_ptr<IClipBackend> CreateDnnBackend(const BackendConfig& cfg)
{
	return std::make_unique<DnnBackend>(cfg);
}

} // namespace ruclip

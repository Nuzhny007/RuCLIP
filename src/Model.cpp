#include "ruclip/Model.h"

#include "ruclip/Backend/ClipMath.h"

#include <stdexcept>

namespace ruclip
{

Model::Model(const ModelOptions& options)
{
	if (options.modelDir.empty())
		throw std::runtime_error("ruclip::Model: modelDir is empty");

	BackendConfig cfg;
	cfg.modelDir = options.modelDir;
	cfg.deviceIndex = options.deviceIndex;
	cfg.maxBatch = options.maxBatch;
	cfg.useFp16 = options.useFp16;
	cfg.verbose = options.verbose;

	backend_ = CreateBackend(options.backend, cfg);
	processor_ = RuCLIPProcessor::FromPretrained(options.modelDir, options.preprocess);
}

TensorF32 Model::EncodeImage(const cv::Mat& image)
{
	return backend_->EncodeImage(processor_.EncodeImage(image));
}

TensorF32 Model::EncodeImage(const std::vector<cv::Mat>& images)
{
	return backend_->EncodeImage(processor_.PrepareImages(images));
}

TensorF32 Model::EncodeText(const std::string& text)
{
	return backend_->EncodeText(processor_.EncodeText(text));
}

TensorF32 Model::EncodeText(const std::vector<std::string>& texts)
{
	return backend_->EncodeText(processor_.PrepareTexts(texts));
}

TensorF32 Model::ProbsPerImage(const std::vector<cv::Mat>& images,
                               const std::vector<std::string>& texts)
{
	const TensorF32 imageFeatures = EncodeImage(images);
	const TensorF32 textFeatures = EncodeText(texts);
	return ruclip::ProbsPerImage(imageFeatures, textFeatures, backend_->logitScale());
}

TensorF32 Model::Relevancy(const std::vector<cv::Mat>& reference,
                           const std::vector<cv::Mat>& candidates,
                           const std::vector<std::string>& texts)
{
	if (reference.size() != 1)
		throw std::runtime_error("ruclip::Model::Relevancy: expected exactly one reference image");

	const TensorF32 embed = L2NormalizeRows(EncodeImage(reference[0]));
	const TensorF32 positives = L2NormalizeRows(EncodeImage(candidates));
	const TensorF32 negatives = L2NormalizeRows(EncodeText(texts));
	return ruclip::Relevancy(embed, positives, negatives);
}

} // namespace ruclip

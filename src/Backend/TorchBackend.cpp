#include "ruclip/Backend/TorchBackend.h"

#include "ruclip/Backend/ClipMath.h"
#include "ruclip/ClipModel.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace ruclip
{
namespace
{

/// torch::Tensor -> TensorF32 (CPU, contiguous, float32).
TensorF32 ToTensorF32(const torch::Tensor& t)
{
	const torch::Tensor c = t.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
	TensorF32 out;
	out.shape.assign(c.sizes().begin(), c.sizes().end());
	const float* p = c.data_ptr<float>();
	out.data.assign(p, p + c.numel());
	return out;
}

/// Обёртка над существующей LibTorch-моделью (CLIPImpl).
/// Принимает нейтральные ImageBatch/TextBatch, возвращает нейтральные TensorF32.
class TorchBackend final : public IClipBackend
{
public:
	explicit TorchBackend(const BackendConfig& cfg)
	{
		// Детерминированная инициализация, как в прежнем main (torch::manual_seed(24)).
		torch::manual_seed(24);

		clip_ = FromPretrained(cfg.modelDir);

		const bool cuda = torch::cuda::is_available();
		device_ = cuda ? torch::Device(torch::kCUDA, cfg.deviceIndex)
		               : torch::Device(torch::kCPU);
		clip_->to(device_);
		clip_->eval();	// инференс, без dropout/BN-статистик

		if (cfg.useFp16)
		{
			if (!device_.is_cuda())
				throw std::runtime_error("TorchBackend: FP16 требует CUDA (на CPU FP16 не поддерживается)");
			clip_->to(torch::kHalf);
			dtype_ = torch::kHalf;
		}

		std::cout << (cuda ? "CUDA is available! Running on GPU."
		                   : "CUDA is not available! Running on CPU.")
		          << (cfg.useFp16 ? " [FP16]" : "") << std::endl;

		// logit_scale — параметр модели; запоминаем exp(logit_scale) один раз.
		torch::NoGradGuard no_grad;
		logitScale_ = clip_->GetLogitScale().to(torch::kFloat32).exp().item<float>();
	}

	BackendKind kind() const override { return BackendKind::Torch; }

	float logitScale() const override { return logitScale_; }

	TensorF32 EncodeImage(const ImageBatch& images) override
	{
		if (images.B <= 0 || images.C <= 0 || images.H <= 0 || images.W <= 0)
			throw std::runtime_error("TorchBackend::EncodeImage: empty batch");

		torch::Tensor t = torch::from_blob(
			const_cast<float*>(images.pixels.data()),
			{ images.B, images.C, images.H, images.W },
			torch::kFloat32).clone();

		return ToTensorF32(clip_->EncodeImage(t.to(device_).to(dtype_)));
	}

	TensorF32 EncodeText(const TextBatch& text) override
	{
		if (text.B <= 0 || text.L <= 0)
			throw std::runtime_error("TorchBackend::EncodeText: empty batch");

		// TextBatch хранит int32; CLIPImpl::EncodeText ожидает kLong (int64).
		torch::Tensor t = torch::from_blob(
			const_cast<int32_t*>(text.ids.data()),
			{ text.B, text.L },
			torch::kInt32).to(torch::kLong);

		return ToTensorF32(clip_->EncodeText(t.to(device_)));
	}

private:
	CLIP clip_{ nullptr };
	torch::Device device_{ torch::kCPU };
	torch::ScalarType dtype_ = torch::kFloat32;
	float logitScale_ = kDefaultLogitScale;
};

} // namespace

std::unique_ptr<IClipBackend> CreateTorchBackend(const BackendConfig& cfg)
{
	return std::make_unique<TorchBackend>(cfg);
}

} // namespace ruclip

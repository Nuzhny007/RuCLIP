#include "Backend/IClipBackend.h"

#include <stdexcept>

#ifdef RUCLIP_HAS_TORCH
#include "Backend/TorchBackend.h"
#endif
#ifdef RUCLIP_HAS_TRT
#include "Backend/TensorRTBackend.h"
#endif
#ifdef RUCLIP_HAS_DNN
#include "Backend/DnnBackend.h"
#endif

namespace ruclip
{

std::unique_ptr<IClipBackend> CreateBackend(BackendKind kind, const BackendConfig& cfg)
{
	switch (kind)
	{
	case BackendKind::Torch:
#ifdef RUCLIP_HAS_TORCH
		return CreateTorchBackend(cfg);
#else
		throw std::runtime_error("RuCLIP: LibTorch backend is not built (RUCLIP_WITH_TORCH=OFF)");
#endif

	case BackendKind::TensorRT:
#ifdef RUCLIP_HAS_TRT
		return CreateTensorRTBackend(cfg);
#else
		throw std::runtime_error("RuCLIP: TensorRT backend is not built (RUCLIP_WITH_TRT=OFF)");
#endif

	case BackendKind::Dnn:
#ifdef RUCLIP_HAS_DNN
		return CreateDnnBackend(cfg);
#else
		throw std::runtime_error("RuCLIP: OpenCV DNN backend is not built (RUCLIP_WITH_DNN=OFF)");
#endif
	}

	throw std::runtime_error("RuCLIP: unknown backend kind");
}

} // namespace ruclip

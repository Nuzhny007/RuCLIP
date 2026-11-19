#include "ruclip/Backend/TensorRTBackend.h"

#include "ruclip/Backend/ClipMath.h"
#include "ruclip/json.hpp"

// Порядок важен: cuda_runtime.h даёт cudaStream_t / cudaError_t для NvInfer.h.
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ruclip
{
namespace
{

void CudaCheck(cudaError_t err, const char* what)
{
	if (err != cudaSuccess)
		throw std::runtime_error(std::string("TensorRT/CUDA error in ") + what + ": " + cudaGetErrorString(err));
}

// В TensorRT 10.16 у интерфейсов удалён destroy(); ownership — через delete.
template <typename T>
using TrtPtr = std::unique_ptr<T>;

class TrtLogger final : public nvinfer1::ILogger
{
public:
	void log(Severity severity, const char* msg) noexcept override
	{
		if (severity <= Severity::kWARNING)
			std::fprintf(stderr, "[TensorRT] %s\n", msg);
	}
};

nvinfer1::Dims MakeDims(std::initializer_list<int64_t> dims)
{
	nvinfer1::Dims d;
	d.nbDims = static_cast<int>(dims.size());
	int i = 0;
	for (int64_t v : dims)
		d.d[i++] = v;
	return d;
}

int64_t Volume(const nvinfer1::Dims& d)
{
	int64_t v = 1;
	for (int i = 0; i < d.nbDims; ++i)
		v *= d.d[i];
	return v;
}

size_t ElemSize(nvinfer1::DataType t)
{
	switch (t)
	{
	case nvinfer1::DataType::kFLOAT: return 4;
	case nvinfer1::DataType::kHALF:  return 2;
	case nvinfer1::DataType::kINT32: return 4;
	case nvinfer1::DataType::kINT64: return 8;
	case nvinfer1::DataType::kUINT8: return 1;
	case nvinfer1::DataType::kBOOL:  return 1;
	default:                         return 4;
	}
}

std::string DimsToString(const nvinfer1::Dims& d)
{
	std::string s = "[";
	for (int i = 0; i < d.nbDims; ++i)
	{
		if (i)
			s += ", ";
		s += std::to_string(d.d[i]);
	}
	s += "]";
	return s;
}

std::vector<char> ReadFile(const std::string& path)
{
	std::ifstream f(path, std::ios::binary | std::ios::ate);
	if (!f)
		throw std::runtime_error("TensorRT: cannot open " + path);
	const std::streamsize size = f.tellg();
	f.seekg(0, std::ios::beg);
	std::vector<char> buf(static_cast<size_t>(size));
	if (!f.read(buf.data(), size))
		throw std::runtime_error("TensorRT: cannot read " + path);
	return buf;
}

void WriteFile(const std::string& path, const void* data, size_t size)
{
	std::ofstream f(path, std::ios::binary | std::ios::trunc);
	if (!f)
		throw std::runtime_error("TensorRT: cannot write " + path);
	f.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
}

/// Имя файла кэша engine: <stem>.<fp32|fp16>.gpu<N>.b<maxBatch>.engine.
/// Все параметры, влияющие на содержимое engine, входят в имя, поэтому файлы
/// разных конфигураций не перетирают друг друга:
///   clip_visual.fp16.gpu0.b8.engine, clip_textual.fp32.gpu1.b1.engine, ...
std::string EngineFileName(const char* stem, bool fp16, int deviceIndex, int maxBatch)
{
	return std::string(stem) + (fp16 ? ".fp16" : ".fp32") +
	       ".gpu" + std::to_string(deviceIndex) +
	       ".b" + std::to_string(maxBatch) + ".engine";
}

/// Один движок (engine) TensorRT: сериализация/десериализация, контекст, буферы.
class TrtEngine
{
public:
	TrtEngine(const std::string& enginePath,
	          const std::string& onnxPath,
	          const std::string& inputName,
	          const std::string& outputName,
	          const nvinfer1::Dims& minDims,
	          const nvinfer1::Dims& optDims,
	          const nvinfer1::Dims& maxDims,
	          bool fp16,
	          TrtLogger& logger,
	          bool verbose)
		: inputName_(inputName), outputName_(outputName)
	{
		std::vector<char> serialized;
		bool fromDisk = false;
		if (std::filesystem::exists(enginePath))
		{
			serialized = ReadFile(enginePath);
			fromDisk = true;
			if (verbose)
				std::fprintf(stderr, "[TensorRT] loaded engine %s (%zu bytes)\n", enginePath.c_str(), serialized.size());
		}

		runtime_.reset(nvinfer1::createInferRuntime(logger));
		if (!runtime_)
			throw std::runtime_error("TensorRT: createInferRuntime failed");

		if (fromDisk)
			engine_.reset(runtime_->deserializeCudaEngine(serialized.data(), serialized.size()));

		if (!engine_)
		{
			// Кэша нет или он несовместим с текущей версией TensorRT / GPU:
			// собираем engine из ONNX и сохраняем рядом с ONNX.
			if (fromDisk && verbose)
				std::fprintf(stderr, "[TensorRT] cached engine is stale, rebuilding %s\n", enginePath.c_str());
			serialized = BuildEngine(onnxPath, inputName, minDims, optDims, maxDims, fp16, logger);
			WriteFile(enginePath, serialized.data(), serialized.size());
			if (verbose)
				std::fprintf(stderr, "[TensorRT] built engine %s (%zu bytes)\n", enginePath.c_str(), serialized.size());
			engine_.reset(runtime_->deserializeCudaEngine(serialized.data(), serialized.size()));
			if (!engine_)
				throw std::runtime_error("TensorRT: failed to deserialize freshly built engine " + enginePath);
		}
		context_.reset(engine_->createExecutionContext());
		if (!context_)
			throw std::runtime_error("TensorRT: createExecutionContext failed");

		inputElemSize_ = ElemSize(engine_->getTensorDataType(inputName_.c_str()));
		outputElemSize_ = ElemSize(engine_->getTensorDataType(outputName_.c_str()));

		if (!context_->setInputShape(inputName_.c_str(), maxDims))
			throw std::runtime_error("TensorRT: setInputShape(max) failed");
		const nvinfer1::Dims outMaxDims = context_->getTensorShape(outputName_.c_str());
		if (outMaxDims.nbDims <= 0)
			throw std::runtime_error("TensorRT: cannot infer output shape for " + outputName);

		maxInputBytes_ = static_cast<size_t>(Volume(maxDims)) * inputElemSize_;
		maxOutputBytes_ = static_cast<size_t>(Volume(outMaxDims)) * outputElemSize_;

		if (verbose)
		{
			std::fprintf(stderr, "[TensorRT] %s: input '%s' max %s, output '%s' max %s\n",
			             enginePath.c_str(), inputName_.c_str(), DimsToString(maxDims).c_str(),
			             outputName_.c_str(), DimsToString(outMaxDims).c_str());
		}

		CudaCheck(cudaMalloc(&dInput_, maxInputBytes_), "cudaMalloc(input)");
		CudaCheck(cudaMalloc(&dOutput_, maxOutputBytes_), "cudaMalloc(output)");
		CudaCheck(cudaStreamCreate(&stream_), "cudaStreamCreate");
	}

	~TrtEngine()
	{
		if (stream_)
			cudaStreamDestroy(stream_);
		if (dInput_)
			cudaFree(dInput_);
		if (dOutput_)
			cudaFree(dOutput_);
	}

	TrtEngine(const TrtEngine&) = delete;
	TrtEngine& operator=(const TrtEngine&) = delete;

	TensorF32 Run(const void* hostInput, size_t inputBytes, const nvinfer1::Dims& inDims)
	{
		if (inputBytes > maxInputBytes_)
			throw std::runtime_error("TensorRT: input exceeds allocated device buffer");

		CudaCheck(cudaMemcpy(dInput_, hostInput, inputBytes, cudaMemcpyHostToDevice), "cudaMemcpy(H2D)");

		if (!context_->setInputShape(inputName_.c_str(), inDims))
			throw std::runtime_error("TensorRT: setInputShape failed");
		if (!context_->setTensorAddress(inputName_.c_str(), dInput_))
			throw std::runtime_error("TensorRT: setTensorAddress(input) failed");
		if (!context_->setTensorAddress(outputName_.c_str(), dOutput_))
			throw std::runtime_error("TensorRT: setTensorAddress(output) failed");

		if (!context_->enqueueV3(stream_))
			throw std::runtime_error("TensorRT: enqueueV3 failed");
		CudaCheck(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");

		const nvinfer1::Dims outDims = context_->getTensorShape(outputName_.c_str());
		if (outDims.nbDims <= 0)
			throw std::runtime_error("TensorRT: invalid output shape");

		TensorF32 out;
		out.shape.assign(outDims.d, outDims.d + outDims.nbDims);
		out.data.resize(static_cast<size_t>(Volume(outDims)));

		CudaCheck(cudaMemcpy(out.data.data(), dOutput_,
		                     out.data.size() * outputElemSize_, cudaMemcpyDeviceToHost),
		          "cudaMemcpy(D2H)");
		return out;
	}

private:
	static std::vector<char> BuildEngine(const std::string& onnxPath,
	                                     const std::string& inputName,
	                                     const nvinfer1::Dims& minDims,
	                                     const nvinfer1::Dims& optDims,
	                                     const nvinfer1::Dims& maxDims,
	                                     bool fp16,
	                                     TrtLogger& logger)
	{
		TrtPtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
		if (!builder)
			throw std::runtime_error("TensorRT: createInferBuilder failed");

		TrtPtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0));
		if (!network)
			throw std::runtime_error("TensorRT: createNetworkV2 failed");

		TrtPtr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
		if (!parser)
			throw std::runtime_error("TensorRT: createParser failed");
		if (!parser->parseFromFile(onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
			throw std::runtime_error("TensorRT: failed to parse ONNX " + onnxPath);

		TrtPtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
		if (!config)
			throw std::runtime_error("TensorRT: createBuilderConfig failed");
		config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, size_t(2) << 30);
		if (fp16)
			config->setFlag(nvinfer1::BuilderFlag::kFP16);

		// У IOptimizationProfile protected-деструктор: владение забирает config
		// при addOptimizationProfile, поэтому оборачивать в unique_ptr нельзя.
		nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
		if (!profile)
			throw std::runtime_error("TensorRT: createOptimizationProfile failed");
		if (!profile->setDimensions(inputName.c_str(), nvinfer1::OptProfileSelector::kMIN, minDims) ||
		    !profile->setDimensions(inputName.c_str(), nvinfer1::OptProfileSelector::kOPT, optDims) ||
		    !profile->setDimensions(inputName.c_str(), nvinfer1::OptProfileSelector::kMAX, maxDims))
			throw std::runtime_error("TensorRT: optimization profile setDimensions failed");
		if (config->addOptimizationProfile(profile) < 0)
			throw std::runtime_error("TensorRT: addOptimizationProfile failed");

		// В TRT 10.16 IHostMemory не имеет destroy(); освобождается через delete.
		std::unique_ptr<nvinfer1::IHostMemory> serialized(builder->buildSerializedNetwork(*network, *config));
		if (!serialized)
			throw std::runtime_error("TensorRT: buildSerializedNetwork failed for " + onnxPath);

		return std::vector<char>(static_cast<const char*>(serialized->data()),
		                         static_cast<const char*>(serialized->data()) + serialized->size());
	}

	std::string inputName_, outputName_;
	TrtPtr<nvinfer1::IRuntime> runtime_;
	TrtPtr<nvinfer1::ICudaEngine> engine_;
	TrtPtr<nvinfer1::IExecutionContext> context_;
	cudaStream_t stream_ = nullptr;
	void* dInput_ = nullptr;
	void* dOutput_ = nullptr;
	size_t inputElemSize_ = 4;
	size_t outputElemSize_ = 4;
	size_t maxInputBytes_ = 0;
	size_t maxOutputBytes_ = 0;
};

class TensorRTBackend final : public IClipBackend
{
public:
	explicit TensorRTBackend(const BackendConfig& cfg)
	{
		using json = nlohmann::json;
		const std::filesystem::path configPath = cfg.modelDir / "config.json";
		std::ifstream f(configPath);
		if (!f)
			throw std::runtime_error("TensorRT: cannot open " + configPath.string());
		json config = json::parse(f);

		imageSize_ = int(config["image_resolution"]);
		contextLength_ = int(config["context_length"]);
		embedDim_ = int(config["embed_dim"]);
		logitScale_ = config.contains("logit_scale")
			? float(config["logit_scale"].get<double>())
			: kDefaultLogitScale;

		const int maxBatch = std::max(1, cfg.maxBatch);

		int deviceCount = 0;
		CudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
		if (deviceCount <= 0)
			throw std::runtime_error("TensorRT: no CUDA device available");
		const int dev = std::min(std::max(0, cfg.deviceIndex), deviceCount - 1);
		CudaCheck(cudaSetDevice(dev), "cudaSetDevice");

		// Кэш engine лежит рядом с ONNX. Имя содержит точность, фактический
		// индекс GPU и maxBatch: все они влияют на содержимое engine.
		const std::string visualOnnx = (cfg.modelDir / "clip_visual.onnx").string();
		const std::string textualOnnx = (cfg.modelDir / "clip_textual.onnx").string();
		if (!std::filesystem::exists(visualOnnx))
			throw std::runtime_error("TensorRT: missing ONNX model " + visualOnnx);
		if (!std::filesystem::exists(textualOnnx))
			throw std::runtime_error("TensorRT: missing ONNX model " + textualOnnx);

		visual_ = std::make_unique<TrtEngine>(
			(cfg.modelDir / EngineFileName("clip_visual", cfg.useFp16, dev, maxBatch)).string(), visualOnnx,
			"pixel_values", "image_features",
			MakeDims({ 1, 3, imageSize_, imageSize_ }),
			MakeDims({ 1, 3, imageSize_, imageSize_ }),
			MakeDims({ maxBatch, 3, imageSize_, imageSize_ }),
			cfg.useFp16, logger_, cfg.verbose);

		textual_ = std::make_unique<TrtEngine>(
			(cfg.modelDir / EngineFileName("clip_textual", cfg.useFp16, dev, maxBatch)).string(), textualOnnx,
			"input_ids", "text_features",
			MakeDims({ 1, contextLength_ }),
			MakeDims({ 1, contextLength_ }),
			MakeDims({ maxBatch, contextLength_ }),
			cfg.useFp16, logger_, cfg.verbose);
	}

	BackendKind kind() const override { return BackendKind::TensorRT; }

	float logitScale() const override { return logitScale_; }

	TensorF32 EncodeImage(const ImageBatch& images) override
	{
		if (images.B <= 0)
			throw std::runtime_error("TensorRTBackend::EncodeImage: empty batch");
		if (images.C != 3 || images.H != imageSize_ || images.W != imageSize_)
			throw std::runtime_error("TensorRTBackend::EncodeImage: unexpected image shape");

		return visual_->Run(images.pixels.data(), images.pixels.size() * sizeof(float),
		                    MakeDims({ images.B, 3, images.H, images.W }));
	}

	TensorF32 EncodeText(const TextBatch& text) override
	{
		if (text.B <= 0)
			throw std::runtime_error("TensorRTBackend::EncodeText: empty batch");
		if (text.L != contextLength_)
			throw std::runtime_error("TensorRTBackend::EncodeText: unexpected sequence length");

		TensorF32 full = textual_->Run(text.ids.data(), text.ids.size() * sizeof(int32_t),
		                               MakeDims({ text.B, text.L }));
		return GatherEos(full, text.eos_pos);
	}

private:
	TrtLogger logger_;
	int imageSize_ = 0;
	int contextLength_ = 0;
	int embedDim_ = 0;
	float logitScale_ = kDefaultLogitScale;
	std::unique_ptr<TrtEngine> visual_;
	std::unique_ptr<TrtEngine> textual_;
};

} // namespace

std::unique_ptr<IClipBackend> CreateTensorRTBackend(const BackendConfig& cfg)
{
	return std::make_unique<TensorRTBackend>(cfg);
}

} // namespace ruclip

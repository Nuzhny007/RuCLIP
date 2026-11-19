// Минимальное приложение-потребитель библиотеки ruCLIP.
//
// Сборка (после `cmake --install` пакета ruCLIP):
//   cmake -S . -B build "-DruCLIP_DIR=<prefix>/lib/cmake/ruCLIP"
//   cmake --build build --config Release
//
// Запуск:
//   ruclip_basic <model_dir> <image> <label1> [label2 ...]
//   env: RUCLIP_BACKEND=torch|trt|dnn (по умолчанию torch), RUCLIP_FP16=1

#include <ruclip/ruclip.h>

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

ruclip::BackendKind ParseBackend(const std::string& name)
{
	if (name == "torch" || name == "libtorch")
		return ruclip::BackendKind::Torch;
	if (name == "trt" || name == "tensorrt")
		return ruclip::BackendKind::TensorRT;
	if (name == "dnn" || name == "opencv")
		return ruclip::BackendKind::Dnn;
	throw std::runtime_error("unknown backend: " + name);
}

} // namespace

int main(int argc, char** argv)
{
	if (argc < 4)
	{
		std::cerr << "usage: ruclip_basic <model_dir> <image> <label1> [label2 ...]\n"
		          << "env: RUCLIP_BACKEND=torch|trt|dnn (default torch), RUCLIP_FP16=1,\n"
		          << "     RUCLIP_MAX_BATCH=<n>, RUCLIP_VERBOSE=1\n";
		return 2;
	}

	try
	{
		ruclip::ModelOptions options;
		options.modelDir = argv[1];
		const char* backendEnv = std::getenv("RUCLIP_BACKEND");
		options.backend = ParseBackend(backendEnv ? backendEnv : "torch");
		options.useFp16 = std::getenv("RUCLIP_FP16") != nullptr;
		options.verbose = std::getenv("RUCLIP_VERBOSE") != nullptr;
		if (const char* batchEnv = std::getenv("RUCLIP_MAX_BATCH"))
			options.maxBatch = std::max(1, std::atoi(batchEnv));

		const cv::Mat image = cv::imread(argv[2], cv::IMREAD_COLOR);
		if (image.empty())
			throw std::runtime_error(std::string("cannot read image: ") + argv[2]);

		const std::vector<std::string> labels(argv + 3, argv + argc);

		ruclip::Model model(options);
		const ruclip::TensorF32 probs = model.ProbsPerImage({ image }, labels);

		for (size_t i = 0; i < labels.size(); ++i)
			std::cout << labels[i] << ": " << probs.At(0, static_cast<int64_t>(i)) << "\n";
	}
	catch (const std::exception& e)
	{
		std::cerr << "error: " << e.what() << "\n";
		return 1;
	}

	return 0;
}

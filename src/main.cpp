#include <algorithm>
#include <chrono>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <memory>
#include <numeric>
#include <regex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

#include "Backend/ClipMath.h"
#include "Backend/IClipBackend.h"
#include "RuCLIPProcessor.h"

namespace
{

using Clock = std::chrono::steady_clock;

double Ms(Clock::time_point a, Clock::time_point b)
{
	return std::chrono::duration<double, std::milli>(b - a).count();
}

const char* BackendName(ruclip::BackendKind kind)
{
	switch (kind)
	{
	case ruclip::BackendKind::Torch:    return "torch";
	case ruclip::BackendKind::TensorRT: return "trt";
	case ruclip::BackendKind::Dnn:      return "dnn";
	}
	return "?";
}

/// Печать матрицы вероятностей [B, N] (аналог прежнего вывода torch::Tensor).
void PrintProbs(const char* title, const ruclip::TensorF32& probs)
{
	std::cout << title << ":\n";
	for (int64_t r = 0; r < probs.rows(); ++r)
	{
		for (int64_t c = 0; c < probs.cols(); ++c)
			std::cout << (c ? "  " : "") << probs.At(r, c);
		std::cout << "\n";
	}
	std::cout.flush();
}

/// Сбор и печать min/avg/max по фазам (в порядке первого добавления).
class TimingCollector
{
public:
	void Add(const std::string& phase, double ms)
	{
		for (auto& p : phases_)
		{
			if (p.name == phase)
			{
				p.values.push_back(ms);
				return;
			}
		}
		phases_.push_back({ phase, { ms } });
	}

	void Print(const char* backend, int repeat, int warmup) const
	{
		// classic-locale: setlocale(LC_ALL, "") иначе печатает десятичную запятую,
		// из-за чего тайминги не парсятся внешними скриптами.
		std::ostringstream out;
		out.imbue(std::locale::classic());
		out << std::fixed << std::setprecision(3);
		out << "\nTiming (backend=" << backend << ", repeats=" << repeat
		    << ", warmup=" << warmup << "):\n";
		for (const auto& p : phases_)
		{
			const double mn = *std::min_element(p.values.begin(), p.values.end());
			const double mx = *std::max_element(p.values.begin(), p.values.end());
			const double sum = std::accumulate(p.values.begin(), p.values.end(), 0.0);
			const double avg = sum / static_cast<double>(p.values.size());
			out << "  " << std::left << std::setw(15) << p.name << std::right
			    << " min " << std::setw(10) << mn
			    << "  avg " << std::setw(10) << avg
			    << "  max " << std::setw(10) << mx << "  ms\n";
		}
		std::cout << out.str();
		std::cout.flush();
	}

private:
	struct Phase
	{
		std::string name;
		std::vector<double> values;
	};
	std::vector<Phase> phases_;
};

} // namespace

///
bool MatchImageAndText(ruclip::IClipBackend& backend, const RuCLIPProcessor& processor,
	const std::vector<cv::Mat>& images, const std::vector<std::string>& labels,
	bool showResult, int repeat)
{
	bool res = false;

	try
	{
		auto runOnce = [&](ruclip::TensorF32& probs) -> std::vector<std::pair<std::string, double>>
		{
			const auto t0 = Clock::now();
			const auto input = processor(labels, images);
			const auto t1 = Clock::now();
			const ruclip::TensorF32 textFeatures = backend.EncodeText(input.first);
			const auto t2 = Clock::now();
			const ruclip::TensorF32 imageFeatures = backend.EncodeImage(input.second);
			const auto t3 = Clock::now();
			probs = ruclip::ProbsPerImage(imageFeatures, textFeatures, backend.logitScale());
			const auto t4 = Clock::now();

			const double pre = Ms(t0, t1);
			const double encT = Ms(t1, t2);
			const double encI = Ms(t2, t3);
			const double post = Ms(t3, t4);
			return {
				{ "preprocess",   pre },
				{ "encode_text",  encT },
				{ "encode_image", encI },
				{ "postprocess",  post },
				{ "total",        pre + encT + encI + post },
			};
		};

		ruclip::TensorF32 probs;
		runOnce(probs);	// прогрев (не учитывается в статистике)

		TimingCollector timing;
		for (int i = 0; i < repeat; ++i)
		{
			for (const auto& [name, ms] : runOnce(probs))
				timing.Add(name, ms);
		}
		timing.Print(BackendName(backend.kind()), repeat, 1);

		PrintProbs("probs per image", probs);

		if (showResult)
		{
			for (size_t row = 0; row < images.size(); ++row)
			{
				const std::string wndName = std::to_string(row);

				float maxProb = 0.f;
				size_t maxCol = 0;
				for (size_t col = 0; col < labels.size(); ++col)
				{
					const float value = probs.At(static_cast<int64_t>(row), static_cast<int64_t>(col));
					if (maxProb < value)
					{
						maxProb = value;
						maxCol = col;
					}
				}
				const std::string label = labels[maxCol] + ": " + std::to_string(maxProb);

				cv::Mat shown = images[row].clone();	// не мутируем входной вектор кадров
				int baseLine = 0;
				const double fontScale = (shown.cols < 1920) ? 0.7 : 2.0;
				const int thickness = (shown.cols < 1920) ? 1 : 2;
				const int lineType = cv::LINE_AA;
				const int fontFace = cv::FONT_HERSHEY_SCRIPT_COMPLEX;
				const cv::Size labelSize = cv::getTextSize(label, fontFace, fontScale, thickness, &baseLine);
				cv::putText(shown, label, cv::Point(2, 2 + labelSize.height), fontFace, fontScale, cv::Scalar(255, 0, 255), thickness, lineType);

				cv::imshow(wndName, shown);
				cv::waitKey(1);
			}
			cv::waitKey(0);
		}

		res = true;
	}
	catch (std::exception& e)
	{
		res = false;
		std::cerr << e.what() << std::endl;
	}
	return res;
}

///
bool MatchOne2ManyImages(ruclip::IClipBackend& backend, const RuCLIPProcessor& processor,
	const std::vector<cv::Mat>& images, const std::vector<std::string>& labels,
	bool showResult, int repeat)
{
	bool res = false;

	if (images.size() < 2)
	{
		std::cerr << "There must be at least 2 images: one for reference and others to search through" << std::endl;
		return res;
	}

	try
	{
		const std::vector<cv::Mat> others(images.begin() + 1, images.end());

		auto runOnce = [&](ruclip::TensorF32& probs) -> std::vector<std::pair<std::string, double>>
		{
			const auto t0 = Clock::now();
			const ruclip::TensorF32 embed = ruclip::L2NormalizeRows(
				backend.EncodeImage(processor.EncodeImage(images[0])));
			const auto t1 = Clock::now();
			const ruclip::TensorF32 imgsFeatures = ruclip::L2NormalizeRows(
				backend.EncodeImage(processor.PrepareImages(others)));
			const auto t2 = Clock::now();
			const ruclip::TensorF32 textFeatures = ruclip::L2NormalizeRows(
				backend.EncodeText(processor.PrepareTexts(labels)));
			const auto t3 = Clock::now();
			probs = ruclip::Relevancy(embed, imgsFeatures, textFeatures);
			const auto t4 = Clock::now();

			const double ref = Ms(t0, t1);
			const double oth = Ms(t1, t2);
			const double txt = Ms(t2, t3);
			const double rel = Ms(t3, t4);
			return {
				{ "reference", ref },
				{ "others",    oth },
				{ "text",      txt },
				{ "relevancy", rel },
				{ "total",     ref + oth + txt + rel },
			};
		};

		ruclip::TensorF32 probs;
		runOnce(probs);	// прогрев (не учитывается в статистике)

		TimingCollector timing;
		for (int i = 0; i < repeat; ++i)
		{
			for (const auto& [name, ms] : runOnce(probs))
				timing.Add(name, ms);
		}
		timing.Print(BackendName(backend.kind()), repeat, 1);

		size_t bestCol = 1;
		float bestProb = -1.f;
		for (size_t col = 1; col < images.size(); ++col)
		{
			const int64_t p = static_cast<int64_t>(col - 1);
			const float prob = probs.Flat(p * 2);	// P(positive)
			std::cout << "  image " << col << ": P(image match) = " << prob << std::endl;
			if (prob > bestProb)
			{
				bestProb = prob;
				bestCol = col;
			}
		}
		std::cout << "Best match: image " << bestCol << " (P = " << bestProb << ")" << std::endl;

		if (showResult)
		{
			cv::imshow("reference", images[0]);
			cv::waitKey(1);

			for (size_t col = 1; col < images.size(); ++col)
			{
				const int64_t p = static_cast<int64_t>(col - 1);
				const float prob = probs.Flat(p * 2);
				const std::string text = "image " + std::to_string(col) + ": " + std::to_string(prob);

				cv::Mat shown = images[col].clone();	// не изменяем входной вектор кадров
				int baseLine = 0;
				const double fontScale = (shown.cols < 1920) ? 0.7 : 2.0;
				const int thickness = (shown.cols < 1920) ? 1 : 2;
				const int lineType = cv::LINE_AA;
				const int fontFace = cv::FONT_HERSHEY_SCRIPT_COMPLEX;
				const cv::Size labelSize = cv::getTextSize(text, fontFace, fontScale, thickness, &baseLine);
				cv::putText(shown, text, cv::Point(2, 2 + labelSize.height), fontFace, fontScale, cv::Scalar(255, 0, 255), thickness, lineType);

				cv::imshow(std::to_string(col), shown);	// отдельное окно на каждого кандидата
				cv::waitKey(1);
			}

			cv::waitKey(0);
		}

		res = true;
	}
	catch (std::exception& e)
	{
		res = false;
		std::cerr << e.what() << std::endl;
	}

	return res;
}


///
int main(int argc, const char* argv[])
{
	setlocale(LC_ALL, "");

	const char* keys =
	{
		"{ test_ind         |0                   | 0: matching images and text, 1: matching first image with all others (text - negative embeddings) | }"
		"{ imgs             |img1.jpg,img2.jpg   | List of images | }"
		"{ text             |cat,bear,fox        | List of labels | }"
		"{ text_file        |                    | UTF-8 file with labels (one per line); overrides --text | }"
		"{ clip             |../data/ruclip-vit-large-patch14-336 | Path to RuClip model | }"
		"{ show             |1                   | Show result | }"
		"{ preprocess       |letterbox           | Image preprocessing: letterbox (default) or centercrop | }"
		"{ backend          |torch               | Inference backend: torch (libtorch) | }"
		"{ max_batch        |8                   | TensorRT: max batch (optimization profile) | }"
		"{ fp16             |0                   | TensorRT: build FP16 engine | }"
		"{ repeat           |1                   | Timing: measured iterations (warmup excluded) | }"
	};

	cv::CommandLineParser parser(argc, argv, keys);
	parser.printMessage();

	const int testInd = parser.get<int>("test_ind");
	const std::string imagesStr = parser.get<std::string>("imgs");
	const std::string labelsStr = parser.get<std::string>("text");
	const std::string textFile = parser.get<std::string>("text_file");
	const std::string pathToClip = parser.get<std::string>("clip");
	const bool showResult = parser.get<int>("show") != 0;
	const std::string backendStr = parser.get<std::string>("backend");
	int repeat = parser.get<int>("repeat");
	if (repeat < 1)
		repeat = 1;

	std::string preprocessStr = parser.get<std::string>("preprocess");
	ImagePreprocess preprocess = ImagePreprocess::Letterbox;
	if (preprocessStr == "centercrop" || preprocessStr == "center-crop" || preprocessStr == "center_crop")
		preprocess = ImagePreprocess::CenterCrop;
	else if (preprocessStr == "stretch")
		preprocess = ImagePreprocess::Stretch;
	else if (preprocessStr != "letterbox")
		std::cerr << "Unknown --preprocess '" << preprocessStr << "', using letterbox" << std::endl;

	ruclip::BackendKind backendKind = ruclip::BackendKind::Torch;
	if (backendStr == "torch" || backendStr == "libtorch")
		backendKind = ruclip::BackendKind::Torch;
	else if (backendStr == "trt" || backendStr == "tensorrt")
		backendKind = ruclip::BackendKind::TensorRT;
	else if (backendStr == "dnn" || backendStr == "opencv" || backendStr == "opencv_dnn")
		backendKind = ruclip::BackendKind::Dnn;
	else
	{
		std::cerr << "Unknown --backend '" << backendStr << "'" << std::endl;
		return 1;
	}

	std::cout << "Load clip from: " << pathToClip << " (backend: " << backendStr << ")" << std::endl;

	ruclip::BackendConfig backendConfig;
	backendConfig.modelDir = pathToClip;
	backendConfig.maxBatch = parser.get<int>("max_batch");
	backendConfig.useFp16 = parser.get<int>("fp16") != 0;

	std::unique_ptr<ruclip::IClipBackend> backend;
	RuCLIPProcessor processor;
	try
	{
		backend = ruclip::CreateBackend(backendKind, backendConfig);

		std::cout << "Image preprocessing: "
		          << (preprocess == ImagePreprocess::CenterCrop ? "centercrop" : "letterbox") << std::endl;
		processor = RuCLIPProcessor::FromPretrained(pathToClip, preprocess);
	}
	catch (const std::exception& e)
	{
		std::cerr << "Failed to create backend/processor from '" << pathToClip << "': " << e.what() << std::endl;
		return 1;
	}

	std::vector<cv::Mat> images;
	{
		std::cout << "images: " << std::endl;
		std::regex sep("[,]+");
		std::sregex_token_iterator tokens(imagesStr.cbegin(), imagesStr.cend(), sep, -1);
		std::sregex_token_iterator end;
		for (; tokens != end; ++tokens)
		{
			cv::Mat img = cv::imread(*tokens, cv::IMREAD_COLOR);
			const cv::Size inputSize(processor.GetImageSize(), processor.GetImageSize());
			std::cout << (*tokens) << " is loaded: " << !img.empty() << " with size " << img.size() << std::endl;
			std::cout << "Resizing to " << inputSize << "..." << std::endl;
			images.emplace_back(img);
		}
	}

	//Завести метки
	std::vector<std::string> labels;
	if (!textFile.empty())
	{
		std::ifstream f(textFile);
		if (!f)
		{
			std::cerr << "Cannot open text file: " << textFile << std::endl;
			return 1;
		}
		std::string line;
		while (std::getline(f, line))
		{
			if (!line.empty() && line.back() == '\r')
				line.pop_back();
			if (!line.empty())
				labels.push_back(line);
		}
		std::cout << "labels (" << textFile << "): ";
		for (const auto& l : labels)
			std::cout << l << " | ";
		std::cout << std::endl;
	}
	else
	{
		std::cout << "labels: ";
		std::regex sep("[,]+");
		std::sregex_token_iterator tokens(labelsStr.cbegin(), labelsStr.cend(), sep, -1);
		std::sregex_token_iterator end;
		for (; tokens != end; ++tokens)
		{
			std::cout << (*tokens) << " | ";
			labels.push_back(*tokens);
		}
		std::cout << std::endl;
	}

	std::cout << "Running..." << std::endl;

	switch (testInd)
	{
	case 0:
		MatchImageAndText(*backend, processor, images, labels, showResult, repeat);
		break;

	case 1:
		MatchOne2ManyImages(*backend, processor, images, labels, showResult, repeat);
		break;

	default:
		std::cerr << "Wrong test index: " << testInd << std::endl;
	}

	std::cout << "The end!" << std::endl;
}

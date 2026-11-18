#include "RuCLIPProcessor.h"
#include "TextLower.h"

///
inline torch::Tensor CVMatToTorchTensor(cv::Mat img, const bool perm = true)
{
	// from_blob не учитывает шаг (step) Mat'а, поэтому для ROI/подматрицы
	// (неcontinuous) он прочитал бы неверные строки. Делаем непрерывную копию.
	if (!img.isContinuous())
		img = img.clone();

	// clone() сразу отвязывает тензор от памяти cv::Mat (защита от изменения/освобождения Mat).
	auto tensor_image = torch::from_blob(img.data, { img.rows, img.cols, img.channels() }, at::kByte).clone();
	if (perm)
		tensor_image = tensor_image.permute({ 2,0,1 });
	tensor_image.unsqueeze_(0);
	tensor_image = tensor_image.toType(c10::kFloat).div(255);
	return tensor_image;		//tensor_image.clone();
}

///
inline cv::Mat TorchTensorToCVMat(const torch::Tensor tensor_image, const bool perm = true)
{
	auto t = tensor_image.detach().to(torch::kCPU);
	// Ранее squeeze() без аргумента убирал ВСЕ размерности 1 (ломая [1,1,H,W] и батчи).
	if (t.dim() == 4 && t.size(0) == 1)
		t = t.squeeze(0);					// убрать только batch=1 -> [C,H,W]
	if (perm && t.dim() == 3)
		t = t.permute({ 1, 2, 0 });			// CHW -> HWC
	t = t.mul(255).clamp(0, 255).to(torch::kU8);
	t = t.contiguous();

	TORCH_CHECK(t.dim() == 2 || t.dim() == 3, "TorchTensorToCVMat: expected HW or HWC tensor");
	const int rows = static_cast<int>(t.size(0));
	const int cols = static_cast<int>(t.size(1));
	const int channels = (t.dim() == 3) ? static_cast<int>(t.size(2)) : 1;

	cv::Mat result_img;
	cv::Mat(rows, cols, CV_MAKETYPE(CV_8U, channels), t.data_ptr()).copyTo(result_img);
	return result_img;
}

///
RuCLIPProcessor :: RuCLIPProcessor(
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
torch::Tensor RuCLIPProcessor :: EncodeText(const/*std::vector<*/std::string &text) const
{
	std::vector<std::vector<int32_t>> ret_ids;
	vkcom::Status status;
	// Как в оригинальном ru-clip: text = text.lower() перед токенизацией.
	// Токенизатор YouTokenToMe регистр сам не понижает, поэтому делаем это здесь.
	const std::string lowered = ToLowerUtf8(text);
	//output_type = vkcom::OutputType::ID, bos = false, eos = false, reverse = false, dropout_prob = 0.0
	std::vector <std::string> texts{ lowered };
	status = Tokenizer->encode_as_ids(texts, &ret_ids);
	if (status.code != 0)
		throw std::runtime_error("RuCLIPProcessor::EncodeText error : " + status.message);
	auto it = ret_ids[0];
	//for (auto &it : ret_ids)
	//{
	if (it.size() > TextSeqLength - 2)
		it.resize(TextSeqLength - 2);
	it.insert(it.begin(), bos_id);			//vector сдвинет при вставке
	it.push_back(eos_id);
	//}
	return PrepareTokens(it);
}

///
cv::Mat RuCLIPProcessor::ResizeToInput(const cv::Mat& img) const
{
	TORCH_CHECK(!img.empty(), "ResizeToInput: empty image");

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
	TORCH_CHECK((ImageSize - newWidth) % 2 == 0 && (ImageSize - newHeight) % 2 == 0,
		"ResizeToInput: padding parity check failed");

	const int xOffset = (ImageSize - newWidth) / 2;
	const int yOffset = (ImageSize - newHeight) / 2;

	TORCH_CHECK(2 * xOffset + newWidth == ImageSize &&
	            2 * yOffset + newHeight == ImageSize,
		"ResizeToInput: centered-padding invariant failed");

	cv::resize(img, newImg(cv::Rect(xOffset, yOffset, newWidth, newHeight)), cv::Size(newWidth, newHeight), 0, 0, cv::INTER_CUBIC);
	return newImg;
}

///
torch::Tensor RuCLIPProcessor::EncodeImage(const cv::Mat& img) const
{
	torch::Tensor img_tensor = CVMatToTorchTensor(ResizeToInput(img), true);
	img_tensor = torch::data::transforms::Normalize<>(NormMean, NormStd)(img_tensor);
	return img_tensor;
}

///
torch::Tensor RuCLIPProcessor::PrepareTokens(/*std::vector<*/std::vector<int32_t> tokens) const //Передаю по значению чтобы внутри иметь дело с копией
{
	// Приводим длину ровно к TextSeqLength, как в оригинальном ru-clip:
	//  - короче  -> дополняем pad_id (0);
	//  - длиннее -> обрезаем, сохраняя последний токен (обычно eos).
	int empty_positions = TextSeqLength - static_cast<int>(tokens.size());
	if (empty_positions > 0)
	{
		tokens.resize(TextSeqLength, static_cast<int32_t>(pad_id));
	}
	else if (empty_positions < 0)
	{
		int32_t back = tokens.back();
		tokens.resize(TextSeqLength);
		tokens.back() = back;
	}

	// После нормализации длина всегда ровно TextSeqLength, поэтому тензор
	// определён во всех случаях. Раньше при empty_positions == 0 (текст ровно
	// в 77 токенов, т.е. от ~75 BPE-токенов) возвращался неопределённый Tensor.
	return torch::tensor(tokens, torch::kLong);
}

///
void RuCLIPProcessor::CacheText(const std::vector <std::string>& texts)
{
	m_textsTensors.clear();
	for (auto& it : texts)
	{
		std::string s = it;
		torch::Tensor text_tensor = EncodeText(s);
		m_textsTensors.push_back(text_tensor);
	}
}

///
const std::vector<torch::Tensor>& RuCLIPProcessor::GetTextTensors() const
{
	return m_textsTensors;
}

///
std::pair<torch::Tensor, torch::Tensor> RuCLIPProcessor::operator()(const std::vector <std::string> &texts, const std::vector <cv::Mat> &images) const
{
	std::vector <torch::Tensor> texts_tensors;
	for (auto& it : texts)
	{
		std::string s = it;
		torch::Tensor text_tensor = EncodeText(s);
		texts_tensors.push_back(text_tensor);
	}

	std::vector <torch::Tensor> images_tensors;
	for (auto &it : images)
	{
		torch::Tensor img_tensor = CVMatToTorchTensor(ResizeToInput(it), true);
		img_tensor = torch::data::transforms::Normalize<>(NormMean, NormStd)(img_tensor);
		//img_tensor.clone();
		images_tensors.push_back(img_tensor);
	}
	return std::make_pair(!texts_tensors.empty()?/*torch::pad_sequence*/torch::stack(texts_tensors):torch::Tensor(), torch::pad_sequence(images_tensors).squeeze(0));
}

///
std::pair<torch::Tensor, torch::Tensor> RuCLIPProcessor::operator()(const std::vector <cv::Mat>& images) const
{
	std::vector <torch::Tensor> images_tensors;
	for (auto& it : images)
	{
		torch::Tensor img_tensor = CVMatToTorchTensor(ResizeToInput(it), true);
		img_tensor = torch::data::transforms::Normalize<>(NormMean, NormStd)(img_tensor);
		//img_tensor.clone();
		images_tensors.push_back(img_tensor);
	}
	return std::make_pair(torch::stack(m_textsTensors), torch::pad_sequence(images_tensors).squeeze(0));
}

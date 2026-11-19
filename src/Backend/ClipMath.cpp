#include "ruclip/Backend/ClipMath.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace ruclip
{
namespace
{

void Check2D(const char* where, const TensorF32& t)
{
	if (t.Dim() != 2)
		throw std::invalid_argument(std::string(where) + ": expected a 2D tensor");
}

/// Вероятности бинарного softmax с температурой 10 для пары (pos, neg).
/// Возвращает {P(positive), P(negative)}.
void BinarySoftmax10(float pos, float neg, float& p0, float& p1)
{
	const float x = 10.f * pos;
	const float y = 10.f * neg;
	const float mx = std::max(x, y);
	const float e0 = std::exp(x - mx);
	const float e1 = std::exp(y - mx);
	const float s = e0 + e1;
	p0 = e0 / s;
	p1 = e1 / s;
}

} // namespace

TensorF32 Transpose2D(const TensorF32& a)
{
	Check2D("Transpose2D", a);
	const int64_t r = a.rows();
	const int64_t c = a.cols();
	TensorF32 out({ c, r });
	for (int64_t i = 0; i < r; ++i)
		for (int64_t j = 0; j < c; ++j)
			out.At(j, i) = a.At(i, j);
	return out;
}

TensorF32 MatMul(const TensorF32& a, const TensorF32& b)
{
	Check2D("MatMul(a)", a);
	Check2D("MatMul(b)", b);
	if (a.cols() != b.rows())
		throw std::invalid_argument("MatMul: inner dimensions do not match");

	const int64_t m = a.rows();
	const int64_t k = a.cols();
	const int64_t n = b.cols();
	TensorF32 out({ m, n });

	for (int64_t i = 0; i < m; ++i)
	{
		float* orow = &out.At(i, 0);
		for (int64_t kk = 0; kk < k; ++kk)
		{
			const float av = a.At(i, kk);
			if (av == 0.f)
				continue;
			const float* brow = &b.At(kk, 0);
			for (int64_t j = 0; j < n; ++j)
				orow[j] += av * brow[j];
		}
	}
	return out;
}

TensorF32 L2NormalizeRows(const TensorF32& a)
{
	Check2D("L2NormalizeRows", a);
	TensorF32 out = a;
	const int64_t r = a.rows();
	const int64_t c = a.cols();

	for (int64_t i = 0; i < r; ++i)
	{
		double sum = 0.0;
		for (int64_t j = 0; j < c; ++j)
		{
			const double v = a.At(i, j);
			sum += v * v;
		}
		const double norm = std::sqrt(sum);
		if (norm > 0.0)
		{
			const float inv = static_cast<float>(1.0 / norm);
			for (int64_t j = 0; j < c; ++j)
				out.At(i, j) = a.At(i, j) * inv;
		}
	}
	return out;
}

TensorF32 SoftmaxRows(const TensorF32& a, float temperature)
{
	Check2D("SoftmaxRows", a);
	TensorF32 out({ a.rows(), a.cols() });
	const int64_t r = a.rows();
	const int64_t c = a.cols();

	for (int64_t i = 0; i < r; ++i)
	{
		float mx = -std::numeric_limits<float>::infinity();
		for (int64_t j = 0; j < c; ++j)
			mx = std::max(mx, a.At(i, j) * temperature);

		double sum = 0.0;
		for (int64_t j = 0; j < c; ++j)
		{
			const double e = std::exp(static_cast<double>(a.At(i, j) * temperature - mx));
			out.At(i, j) = static_cast<float>(e);
			sum += e;
		}
		if (sum > 0.0)
		{
			for (int64_t j = 0; j < c; ++j)
				out.At(i, j) = static_cast<float>(out.At(i, j) / sum);
		}
	}
	return out;
}

TensorF32 GatherRows(const TensorF32& a, const std::vector<int32_t>& rows)
{
	if (a.Dim() < 1)
		throw std::invalid_argument("GatherRows: expected at least 1D tensor");

	const int64_t n = a.rows();
	const int64_t trailing = (n > 0) ? a.Numel() / n : 0;

	std::vector<int64_t> newShape = a.shape;
	newShape[0] = static_cast<int64_t>(rows.size());
	TensorF32 out(newShape);

	for (size_t r = 0; r < rows.size(); ++r)
	{
		const int32_t src = rows[r];
		if (src < 0 || src >= n)
			throw std::out_of_range("GatherRows: row index out of range");
		const float* srcPtr = a.Data() + static_cast<int64_t>(src) * trailing;
		float* dstPtr = out.Data() + static_cast<int64_t>(r) * trailing;
		std::copy(srcPtr, srcPtr + trailing, dstPtr);
	}
	return out;
}

TensorF32 Logits(const TensorF32& imageFeatures, const TensorF32& textFeatures, float logitScale)
{
	Check2D("Logits(imageFeatures)", imageFeatures);
	Check2D("Logits(textFeatures)", textFeatures);
	if (imageFeatures.cols() != textFeatures.cols())
		throw std::invalid_argument("Logits: feature dims do not match");

	const TensorF32 img = L2NormalizeRows(imageFeatures);
	const TensorF32 txt = L2NormalizeRows(textFeatures);
	TensorF32 sims = MatMul(img, Transpose2D(txt));
	for (float& v : sims.data)
		v *= logitScale;
	return sims;
}

TensorF32 ProbsPerImage(const TensorF32& imageFeatures, const TensorF32& textFeatures, float logitScale)
{
	return SoftmaxRows(Logits(imageFeatures, textFeatures, logitScale));
}

TensorF32 Relevancy(const TensorF32& embeds, const TensorF32& positives, const TensorF32& negatives)
{
	Check2D("Relevancy(embeds)", embeds);
	Check2D("Relevancy(positives)", positives);
	Check2D("Relevancy(negatives)", negatives);

	const int64_t B = embeds.rows();
	const int64_t D = embeds.cols();
	const int64_t P = positives.rows();
	const int64_t M = negatives.rows();

	if (P < 1 || M < 1)
		throw std::invalid_argument("Relevancy: need at least one positive and one negative");
	if (positives.cols() != D || negatives.cols() != D)
		throw std::invalid_argument("Relevancy: feature dims do not match");

	// all = [positives; negatives] -> [P + M, D]
	TensorF32 all({ P + M, D });
	std::copy(positives.Data(), positives.Data() + P * D, all.Data());
	std::copy(negatives.Data(), negatives.Data() + M * D, all.Data() + P * D);

	TensorF32 logits = MatMul(embeds, Transpose2D(all)); // [B, P + M]
	TensorF32 out({ B, P, 2 });

	for (int64_t b = 0; b < B; ++b)
	{
		for (int64_t p = 0; p < P; ++p)
		{
			const float pos = logits.At(b, p);

			// Ищем негатив, для которого P(positive) минимальна.
			int64_t bestM = 0;
			float bestPosProb = std::numeric_limits<float>::infinity();
			for (int64_t m = 0; m < M; ++m)
			{
				float p0 = 0.f, p1 = 0.f;
				BinarySoftmax10(pos, logits.At(b, P + m), p0, p1);
				if (p0 < bestPosProb)
				{
					bestPosProb = p0;
					bestM = m;
				}
			}

			float p0 = 0.f, p1 = 0.f;
			BinarySoftmax10(pos, logits.At(b, P + bestM), p0, p1);
			const size_t idx = static_cast<size_t>((b * P + p) * 2);
			out.data[idx] = p0;
			out.data[idx + 1] = p1;
		}
	}
	return out;
}

TensorF32 GatherEos(const TensorF32& fullSequence, const std::vector<int32_t>& eosPos)
{
	if (fullSequence.Dim() != 3)
		throw std::invalid_argument("GatherEos: expected a [B, L, D] tensor");

	const int64_t B = fullSequence.shape[0];
	const int64_t L = fullSequence.shape[1];
	const int64_t D = fullSequence.shape[2];
	if (static_cast<int64_t>(eosPos.size()) != B)
		throw std::invalid_argument("GatherEos: eosPos size must equal B");

	TensorF32 out({ B, D });
	for (int64_t b = 0; b < B; ++b)
	{
		const int32_t pos = eosPos[static_cast<size_t>(b)];
		if (pos < 0 || pos >= L)
			throw std::out_of_range("GatherEos: eos position out of range");
		const float* src = fullSequence.Data() + (static_cast<int64_t>(b) * L + pos) * D;
		std::copy(src, src + D, out.Data() + b * D);
	}
	return out;
}

std::vector<float> GatherEos(const std::vector<float>& full, int B, int L, int D,
                             const std::vector<int32_t>& eosPos)
{
	if (B < 0 || L < 0 || D < 0)
		throw std::invalid_argument("GatherEos: negative size");
	if (static_cast<int64_t>(full.size()) != static_cast<int64_t>(B) * L * D)
		throw std::invalid_argument("GatherEos: full.size() does not match [B, L, D]");
	if (static_cast<int>(eosPos.size()) != B)
		throw std::invalid_argument("GatherEos: eosPos size must equal B");

	std::vector<float> out(static_cast<size_t>(B) * static_cast<size_t>(D));
	for (int b = 0; b < B; ++b)
	{
		const int pos = eosPos[static_cast<size_t>(b)];
		if (pos < 0 || pos >= L)
			throw std::out_of_range("GatherEos: eos position out of range");
		const float* src = full.data() + (static_cast<int64_t>(b) * L + pos) * D;
		std::copy(src, src + D, out.data() + static_cast<int64_t>(b) * D);
	}
	return out;
}

} // namespace ruclip

// Standalone-тест нейтрального слоя (Фаза 0): не требует LibTorch/OpenCV.
//
// Сборка и запуск (пример):
//   cmake -S tests -B tests/build -G "Visual Studio 17 2022" -A x64
//   cmake --build tests/build --config Release
//   tests/build/Release/test_clip_math.exe

#include "Backend/ClipMath.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace
{

int g_failures = 0;

#define CHECK(cond)                                                              \
	do                                                                           \
	{                                                                            \
		if (!(cond))                                                             \
		{                                                                        \
			std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);          \
			++g_failures;                                                        \
		}                                                                        \
	} while (0)

bool Near(float a, float b, float eps = 1e-5f)
{
	return std::fabs(a - b) <= eps;
}

void TestMatMul()
{
	ruclip::TensorF32 a({ 2, 3 });
	a.data = { 1, 2, 3, 4, 5, 6 };
	ruclip::TensorF32 b({ 3, 2 });
	b.data = { 7, 8, 9, 10, 11, 12 };

	const ruclip::TensorF32 c = ruclip::MatMul(a, b);
	CHECK(c.rows() == 2 && c.cols() == 2);
	CHECK(Near(c.At(0, 0), 58.f));
	CHECK(Near(c.At(0, 1), 64.f));
	CHECK(Near(c.At(1, 0), 139.f));
	CHECK(Near(c.At(1, 1), 154.f));
}

void TestTranspose()
{
	ruclip::TensorF32 a({ 2, 3 });
	a.data = { 1, 2, 3, 4, 5, 6 };
	const ruclip::TensorF32 t = ruclip::Transpose2D(a);
	CHECK(t.rows() == 3 && t.cols() == 2);
	CHECK(Near(t.At(0, 0), 1.f) && Near(t.At(0, 1), 4.f));
	CHECK(Near(t.At(1, 0), 2.f) && Near(t.At(1, 1), 5.f));
	CHECK(Near(t.At(2, 0), 3.f) && Near(t.At(2, 1), 6.f));
}

void TestL2NormalizeRows()
{
	ruclip::TensorF32 a({ 3, 2 });
	a.data = { 3, 4, 0, 0, 1, -1 };
	const ruclip::TensorF32 n = ruclip::L2NormalizeRows(a);
	CHECK(Near(n.At(0, 0), 0.6f) && Near(n.At(0, 1), 0.8f));
	CHECK(Near(n.At(1, 0), 0.f) && Near(n.At(1, 1), 0.f));
	CHECK(Near(n.At(2, 0), 0.70710678f, 1e-5f) && Near(n.At(2, 1), -0.70710678f, 1e-5f));
}

void TestSoftmaxRows()
{
	ruclip::TensorF32 a({ 2, 3 });
	a.data = { 1, 2, 3, 0, 0, 0 };
	const ruclip::TensorF32 s = ruclip::SoftmaxRows(a);
	CHECK(Near(s.At(0, 0) + s.At(0, 1) + s.At(0, 2), 1.f));
	CHECK(s.At(0, 2) > s.At(0, 1) && s.At(0, 1) > s.At(0, 0));
	CHECK(Near(s.At(1, 0), 1.f / 3.f) && Near(s.At(1, 1), 1.f / 3.f) && Near(s.At(1, 2), 1.f / 3.f));
}

void TestLogitsIdentity()
{
	// Единичные строки уже L2-нормализованы: при scale=1 матрица не меняется.
	ruclip::TensorF32 img({ 2, 2 });
	img.data = { 1, 0, 0, 1 };
	ruclip::TensorF32 txt({ 2, 2 });
	txt.data = { 1, 0, 0, 1 };

	const ruclip::TensorF32 logits = ruclip::Logits(img, txt, 1.f);
	CHECK(logits.rows() == 2 && logits.cols() == 2);
	CHECK(Near(logits.At(0, 0), 1.f) && Near(logits.At(0, 1), 0.f));
	CHECK(Near(logits.At(1, 0), 0.f) && Near(logits.At(1, 1), 1.f));

	// По умолчанию используется kDefaultLogitScale = 1/0.07.
	const ruclip::TensorF32 scaled = ruclip::Logits(img, txt);
	CHECK(Near(scaled.At(0, 0), ruclip::kDefaultLogitScale, 1e-4f));
}

void TestProbsPerImage()
{
	ruclip::TensorF32 img({ 1, 2 });
	img.data = { 1, 0 };
	ruclip::TensorF32 txt({ 2, 2 });
	txt.data = { 1, 0, 0, 1 };

	const ruclip::TensorF32 probs = ruclip::ProbsPerImage(img, txt, 1.f);
	CHECK(Near(probs.At(0, 0) + probs.At(0, 1), 1.f));
	CHECK(probs.At(0, 0) > probs.At(0, 1));
}

void TestRelevancyEqual()
{
	// D=1: sims = [1(pos), 1(neg0), 0(neg1)].
	// Для m=0: x=y=10 -> P(pos)=0.5; для m=1: P(pos)≈0.99995.
	// Самый сложный негатив — m=0, результат [0.5, 0.5].
	ruclip::TensorF32 embeds({ 1, 1 });
	embeds.data = { 1.f };
	ruclip::TensorF32 positives({ 1, 1 });
	positives.data = { 1.f };
	ruclip::TensorF32 negatives({ 2, 1 });
	negatives.data = { 1.f, 0.f };

	const ruclip::TensorF32 r = ruclip::Relevancy(embeds, positives, negatives);
	CHECK(r.Dim() == 3 && r.shape[0] == 1 && r.shape[1] == 1 && r.shape[2] == 2);
	CHECK(Near(r.Flat(0), 0.5f, 1e-6f));
	CHECK(Near(r.Flat(1), 0.5f, 1e-6f));
}

void TestRelevancyOrthogonal()
{
	// D=2: sim(pos)=1, sim(neg)=0 -> x=10, y=0.
	// P(pos) = 1 / (1 + e^-10) ≈ 0.9999546.
	ruclip::TensorF32 embeds({ 1, 2 });
	embeds.data = { 1.f, 0.f };
	ruclip::TensorF32 positives({ 1, 2 });
	positives.data = { 1.f, 0.f };
	ruclip::TensorF32 negatives({ 1, 2 });
	negatives.data = { 0.f, 1.f };

	const ruclip::TensorF32 r = ruclip::Relevancy(embeds, positives, negatives);
	CHECK(Near(r.Flat(0), 0.9999546f, 1e-6f));
	CHECK(Near(r.Flat(0) + r.Flat(1), 1.f, 1e-6f));
}

void TestRelevancyBatchPositive()
{
	// B=1, P=2, M=1: проверяем форму [B, P, 2] и что она не падает при P>1
	// (это ровно тот случай, из-за которого падала исходная реализация).
	ruclip::TensorF32 embeds({ 1, 2 });
	embeds.data = { 1.f, 0.f };
	ruclip::TensorF32 positives({ 2, 2 });
	positives.data = { 1.f, 0.f, 0.f, 1.f };
	ruclip::TensorF32 negatives({ 1, 2 });
	negatives.data = { 0.5f, 0.5f };

	const ruclip::TensorF32 r = ruclip::Relevancy(embeds, positives, negatives);
	CHECK(r.Dim() == 3 && r.shape[0] == 1 && r.shape[1] == 2 && r.shape[2] == 2);
	CHECK(Near(r.Flat(0) + r.Flat(1), 1.f, 1e-6f));
	CHECK(Near(r.Flat(2) + r.Flat(3), 1.f, 1e-6f));
}

void TestGatherEosTensor()
{
	ruclip::TensorF32 full({ 2, 3, 2 });
	full.data = {  0, 0,  1, 1,  2, 2,
	              10, 10, 11, 11, 12, 12 };
	const std::vector<int32_t> eos = { 1, 2 };
	const ruclip::TensorF32 out = ruclip::GatherEos(full, eos);
	CHECK(out.rows() == 2 && out.cols() == 2);
	CHECK(Near(out.At(0, 0), 1.f) && Near(out.At(0, 1), 1.f));
	CHECK(Near(out.At(1, 0), 12.f) && Near(out.At(1, 1), 12.f));
}

void TestGatherEosFlat()
{
	const std::vector<float> full = { 0, 0, 1, 1, 2, 2, 10, 10, 11, 11, 12, 12 };
	const std::vector<int32_t> eos = { 1, 2 };
	const std::vector<float> out = ruclip::GatherEos(full, 2, 3, 2, eos);
	CHECK(out.size() == 4);
	CHECK(Near(out[0], 1.f) && Near(out[1], 1.f));
	CHECK(Near(out[2], 12.f) && Near(out[3], 12.f));
}

} // namespace

int main()
{
	TestMatMul();
	TestTranspose();
	TestL2NormalizeRows();
	TestSoftmaxRows();
	TestLogitsIdentity();
	TestProbsPerImage();
	TestRelevancyEqual();
	TestRelevancyOrthogonal();
	TestRelevancyBatchPositive();
	TestGatherEosTensor();
	TestGatherEosFlat();

	if (g_failures == 0)
	{
		std::printf("OK: all ClipMath tests passed\n");
		return 0;
	}
	std::printf("FAILED: %d check(s)\n", g_failures);
	return 1;
}

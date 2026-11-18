#pragma once

#include "Backend/IClipBackend.h"

namespace ruclip
{

/// Фабрика TensorRT-бэкенда (определена в TensorRTBackend.cpp, собирается
/// только при RUCLIP_WITH_TRT=ON).
std::unique_ptr<IClipBackend> CreateTensorRTBackend(const BackendConfig& cfg);

} // namespace ruclip

#pragma once

#include "Backend/IClipBackend.h"

namespace ruclip
{

/// Фабрика LibTorch-бэкенда (определена в TorchBackend.cpp, собирается
/// только при RUCLIP_WITH_TORCH=ON).
std::unique_ptr<IClipBackend> CreateTorchBackend(const BackendConfig& cfg);

} // namespace ruclip

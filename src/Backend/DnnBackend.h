#pragma once

#include "Backend/IClipBackend.h"

namespace ruclip
{

/// Фабрика OpenCV DNN-бэкенда (определена в DnnBackend.cpp, собирается
/// только при RUCLIP_WITH_DNN=ON; требует OpenCV 5.x).
std::unique_ptr<IClipBackend> CreateDnnBackend(const BackendConfig& cfg);

} // namespace ruclip

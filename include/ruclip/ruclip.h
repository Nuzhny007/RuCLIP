#pragma once

/// Umbrella-заголовок публичного API библиотеки ruclip.
///
///   #include <ruclip/ruclip.h>
///
/// Достаточно для высокоуровневого использования через ruclip::Model,
/// а также для низкоуровневой сборки процессора и бэкенда вручную.

#include "ruclip/Backend/ClipMath.h"
#include "ruclip/Backend/ClipTypes.h"
#include "ruclip/Backend/IClipBackend.h"
#include "ruclip/Model.h"
#include "ruclip/RuCLIPProcessor.h"

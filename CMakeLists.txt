# CMakeLists.txt: проект CMake для RuCLIP (статическая библиотека + CLI).
#
# Библиотека:  ruclip  (alias ruclip::ruclip) — нейтральный слой, процессор,
#              бэкенды инференса и высокоуровневый фасад ruclip::Model.
# CLI:         RuCLIP  — утилита демонстрации (src/main.cpp).
# Пакет:       ruCLIP  — find_package(ruCLIP) после `cmake --install`.
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(RuCLIP VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)

option(RUCLIP_BUILD_CLI   "Собрать CLI-утилиту RuCLIP" ON)
option(RUCLIP_BUILD_TESTS "Собрать unit-тесты ClipMath" OFF)

# --- Бэкенды ---
option(RUCLIP_WITH_TORCH "Собрать бэкенд LibTorch"   ON)
option(RUCLIP_WITH_TRT   "Собрать бэкенд TensorRT"   OFF)
option(RUCLIP_WITH_DNN   "Собрать бэкенд OpenCV DNN" OFF)

if(NOT RUCLIP_WITH_TORCH AND NOT RUCLIP_WITH_TRT AND NOT RUCLIP_WITH_DNN)
  message(FATAL_ERROR "RuCLIP: включите хотя бы один бэкенд (RUCLIP_WITH_TORCH / RUCLIP_WITH_TRT / RUCLIP_WITH_DNN)")
endif()

# --- OpenCV ---
# Библиотеке нужны core/imgproc (cv::Mat в публичном API), DNN-бэкенду — dnn,
# CLI — imgcodecs (imread) и highgui (imshow).
set(RUCLIP_LIB_OPENCV_COMPONENTS core imgproc)
if(RUCLIP_WITH_DNN)
  list(APPEND RUCLIP_LIB_OPENCV_COMPONENTS dnn)
endif()

set(_ruclip_opencv_components ${RUCLIP_LIB_OPENCV_COMPONENTS})
if(RUCLIP_BUILD_CLI)
  list(APPEND _ruclip_opencv_components imgcodecs highgui)
endif()

find_package(OpenCV REQUIRED COMPONENTS ${_ruclip_opencv_components})
if(OpenCV_VERSION VERSION_LESS "4.6")
  message(FATAL_ERROR "RuCLIP: требуется OpenCV >= 4.6 (найдено ${OpenCV_VERSION})")
endif()
# DNN-бэкенд использует новый graph-движок и нативный LayerNorm — нужен OpenCV 5.x.
if(RUCLIP_WITH_DNN AND OpenCV_VERSION VERSION_LESS "5.0")
  message(FATAL_ERROR "RuCLIP: OpenCV DNN backend требует OpenCV 5.x (найдено ${OpenCV_VERSION})")
endif()

# torch (cu126) ожидает цель CUDA::nvToolsExt, которой нет в CUDA 13
# (NVTX3 заменил nvToolsExt). Библиотека приложению не нужна (см. README),
# поэтому достаточно пустой INTERFACE-цели, чтобы CMake прошёл генерацию.
if(NOT TARGET CUDA::nvToolsExt)
  add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
endif()

if(RUCLIP_WITH_TORCH)
  find_package(Torch REQUIRED)
endif()

if(RUCLIP_WITH_TRT)
  set(RUCLIP_TRT_ROOT "" CACHE PATH "TensorRT root (содержит include/ и lib/)")
  if(NOT RUCLIP_TRT_ROOT)
    if(DEFINED ENV{TENSORRT_ROOT})
      set(RUCLIP_TRT_ROOT "$ENV{TENSORRT_ROOT}")
    else()
      file(GLOB _trt_candidates "C:/work/libraries/cuda/TensorRT-*")
      list(LENGTH _trt_candidates _trt_count)
      if(_trt_count GREATER 0)
        list(SORT _trt_candidates)
        math(EXPR _trt_last "${_trt_count} - 1")
        list(GET _trt_candidates ${_trt_last} RUCLIP_TRT_ROOT)
      endif()
    endif()
  endif()

  find_path(RUCLIP_NVINFER_INCLUDE_DIR NvInfer.h
            HINTS "${RUCLIP_TRT_ROOT}/include" REQUIRED)
  find_library(RUCLIP_NVINFER_LIB nvinfer_10
               HINTS "${RUCLIP_TRT_ROOT}/lib" REQUIRED)
  find_library(RUCLIP_NVONNXPARSER_LIB nvonnxparser_10
               HINTS "${RUCLIP_TRT_ROOT}/lib" REQUIRED)
  message(STATUS "RuCLIP: TensorRT root = ${RUCLIP_TRT_ROOT}")

  find_package(CUDAToolkit REQUIRED)
endif()

include("CMakeLists.Files.txt")

# ---------------------------------------------------------------------------
# Библиотека
# ---------------------------------------------------------------------------
add_library(ruclip STATIC ${SOURCES} ${HEADERS})
add_library(ruclip::ruclip ALIAS ruclip)

target_include_directories(ruclip
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# OpenCV — публично: cv::Mat участвует в публичном API (RuCLIPProcessor.h).
target_link_libraries(ruclip PUBLIC opencv_core opencv_imgproc)

if(RUCLIP_WITH_TORCH)
  target_compile_definitions(ruclip PRIVATE RUCLIP_HAS_TORCH=1)
  # LibTorch — только в реализации, потребителю не нужны его инклуды.
  target_link_libraries(ruclip PRIVATE ${TORCH_LIBRARIES})
endif()

if(RUCLIP_WITH_TRT)
  target_compile_definitions(ruclip PRIVATE RUCLIP_HAS_TRT=1)
  target_include_directories(ruclip PRIVATE ${RUCLIP_NVINFER_INCLUDE_DIR})
  target_link_libraries(ruclip PRIVATE
    ${RUCLIP_NVINFER_LIB}
    ${RUCLIP_NVONNXPARSER_LIB}
    CUDA::cudart)
endif()

if(RUCLIP_WITH_DNN)
  target_compile_definitions(ruclip PRIVATE RUCLIP_HAS_DNN=1)
  target_link_libraries(ruclip PRIVATE opencv_dnn)
endif()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if(RUCLIP_BUILD_CLI)
  # Имя цели должно отличаться от цели библиотеки не только регистром:
  # на Windows MSBuild разрешает *.vcxproj без учёта регистра, и RuCLIP/ruclip
  # дали бы самоссылку (MSB4006). Исполняемый файл по-прежнему RuCLIP.exe.
  add_executable(ruclip_cli src/main.cpp)
  # Import-library exe-цели тоже называется RuCLIP.lib и на Windows совпадает
  # по имени с ruclip.lib статической библиотеки (LNK1149). Уводим её в
  # отдельный каталог, сохраняя имена обоих артефактов.
  set_target_properties(ruclip_cli PROPERTIES
    OUTPUT_NAME RuCLIP
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/implib")
  target_link_libraries(ruclip_cli PRIVATE ruclip opencv_imgcodecs opencv_highgui)

  if(MSVC AND RUCLIP_WITH_TORCH)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET ruclip_cli
                       POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${TORCH_DLLS}
                       $<TARGET_FILE_DIR:ruclip_cli>)
  endif()
endif()

if(MSVC)
  target_compile_options(ruclip PRIVATE /bigobj /openmp /utf-8)
  if(TARGET ruclip_cli)
    target_compile_options(ruclip_cli PRIVATE /bigobj /openmp /utf-8)
  endif()
endif()

if(MSVC_IDE)
  source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES} ${HEADERS})
endif()

# ---------------------------------------------------------------------------
# Тесты (отдельная цель, не входит в установку)
# ---------------------------------------------------------------------------
if(RUCLIP_BUILD_TESTS)
  enable_testing()
  add_executable(test_clip_math tests/test_clip_math.cpp src/Backend/ClipMath.cpp)
  target_include_directories(test_clip_math PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_test(NAME clip_math COMMAND test_clip_math)
endif()

# ---------------------------------------------------------------------------
# Установка и экспорт пакета ruCLIP
# ---------------------------------------------------------------------------
install(TARGETS ruclip
        EXPORT ruCLIPTargets
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

# В сборке без LibTorch заголовки модели/обёртки Torch не устанавливаем:
# они включают <torch/torch.h> и в такой пакет неприменимы.
set(_ruclip_install_excludes)
if(NOT RUCLIP_WITH_TORCH)
  list(APPEND _ruclip_install_excludes
    PATTERN "ClipModel.h" EXCLUDE
    PATTERN "TorchHeader.h" EXCLUDE)
endif()

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} ${_ruclip_install_excludes})

install(EXPORT ruCLIPTargets
        NAMESPACE ruclip::
        FILE ruCLIPTargets.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ruCLIP)

include(CMakePackageConfigHelpers)

set(RUCLIP_CMAKE_CONFIG_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ruCLIP)

# В шаблон подставляем компоненты через пробел, чтобы find_dependency получил
# их как отдельные аргументы без неоднозначности со списком через ';'.
string(REPLACE ";" " " RUCLIP_LIB_OPENCV_COMPONENTS_SPACED "${RUCLIP_LIB_OPENCV_COMPONENTS}")

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ruCLIPConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/ruCLIPConfig.cmake"
  INSTALL_DESTINATION "${RUCLIP_CMAKE_CONFIG_DESTINATION}")

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/ruCLIPConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/ruCLIPConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/ruCLIPConfigVersion.cmake"
  DESTINATION "${RUCLIP_CMAKE_CONFIG_DESTINATION}")
